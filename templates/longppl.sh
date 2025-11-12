#!/bin/bash
#SBATCH --job-name={{ slurm_config.name }}
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --cpus-per-task=32

#SBATCH --output={{ slurm_config.log_dir }}/%j.out
#SBATCH --error={{ slurm_config.log_dir }}/%j.err

#SBATCH --account={{ slurm_config.account }}
#SBATCH --partition={{ slurm_config.partition }}
#SBATCH --gres={{ slurm_config.gres }}
#SBATCH --time={{ slurm_config.time }}

# link latest log files
ln -sf {{ slurm_config.log_dir }}/$SLURM_JOB_ID.out {{ slurm_config.log_dir }}/latest.out
ln -sf {{ slurm_config.log_dir }}/$SLURM_JOB_ID.err {{ slurm_config.log_dir }}/latest.err

set -euo pipefail

export IMG="/scratch/{{ slurm_config.account }}/containers/vllm_v10.1.1.sif"
export SCR="$PWD"                          # SCR = scratch directory, will be /workspace in container
export ACC="{{ slurm_config.account }}"

# Determine which project's scratch to mount as /project
# If MODEL starts with /scratch/project_X, use that project; otherwise use current project
MODEL_PATH="{{ env_vars.MODEL }}"
if [[ "$MODEL_PATH" =~ ^/scratch/(project_[0-9]+)/ ]]; then
  MODEL_PROJECT="${BASH_REMATCH[1]}"
  export PRJ="/scratch/$MODEL_PROJECT"
  echo "Detected model in $MODEL_PROJECT, binding /scratch/$MODEL_PROJECT to /project"
else
  export PRJ="/scratch/{{ slurm_config.account }}"
  echo "Model not in different project scratch, using current project"
fi

# Parse gres for GPU count (e.g., "gpu:mi250:4" -> 4)
GRES="{{ slurm_config.gres }}"
if [[ "$GRES" =~ gpu:[^:]*:([0-9]+) ]]; then
    GPUS="${BASH_REMATCH[1]}"
elif [[ "$GRES" =~ gpu:([0-9]+) ]]; then
    GPUS="${BASH_REMATCH[1]}"
else
    echo "Warning: Could not parse GPU count from GRES '$GRES', defaulting to 2"
    GPUS=2  # LongPPL default: 2 GPUs sufficient for most models
fi

# topology & model knobs
export N_NODES=1
export MODEL_ID="{{ env_vars.MODEL }}"

# Rewrite model path for container: /scratch/project_X/... -> /project/...
if [[ "$MODEL_ID" =~ ^/scratch/project_[0-9]+/(.*)$ ]]; then
  MODEL_ID_CONTAINER="/project/${BASH_REMATCH[1]}"
  echo "Rewriting model path for container: $MODEL_ID -> $MODEL_ID_CONTAINER"
  export MODEL_ID="$MODEL_ID_CONTAINER"
fi

# Rewrite dataset path for container if provided
{% if env_vars.LONGPPL_DATASET %}DATASET_PATH="{{ env_vars.LONGPPL_DATASET }}"
# Convert relative paths to absolute (relative to $SCR)
if [[ ! "$DATASET_PATH" =~ ^/ ]]; then
  DATASET_PATH="$SCR/$DATASET_PATH"
  echo "Converting relative dataset path to absolute: $DATASET_PATH"
fi

# Normalize paths (remove ./ and ../)
DATASET_PATH=$(realpath -m "$DATASET_PATH")
SCR_NORMALIZED=$(realpath -m "$SCR")
echo "Normalized dataset path: $DATASET_PATH"
echo "Normalized SCR: $SCR_NORMALIZED"

# Rewrite absolute paths for container
# First check if dataset is under workspace directory (most common case for relative paths)
if [[ "$DATASET_PATH" =~ ^${SCR_NORMALIZED}/(.*)$ ]]; then
  # Path is under $SCR (the current scratch dir), map to /workspace
  RELATIVE_PATH="${BASH_REMATCH[1]}"
  DATASET_PATH_CONTAINER="/workspace/$RELATIVE_PATH"
  echo "Rewriting dataset path for container (workspace): $DATASET_PATH -> $DATASET_PATH_CONTAINER"
  export DATASET_PATH="$DATASET_PATH_CONTAINER"
# Handle /pfs/lustrep2/scratch/project_X/... pattern (LUMI normalized paths)
elif [[ "$DATASET_PATH" =~ ^/pfs/lustrep2/scratch/project_[0-9]+/(.*)$ ]]; then
  DATASET_PATH_CONTAINER="/project/${BASH_REMATCH[1]}"
  echo "Rewriting dataset path for container (project): $DATASET_PATH -> $DATASET_PATH_CONTAINER"
  export DATASET_PATH="$DATASET_PATH_CONTAINER"
# Handle /scratch/project_X/... pattern (standard paths)
elif [[ "$DATASET_PATH" =~ ^/scratch/project_[0-9]+/(.*)$ ]]; then
  DATASET_PATH_CONTAINER="/project/${BASH_REMATCH[1]}"
  echo "Rewriting dataset path for container (project): $DATASET_PATH -> $DATASET_PATH_CONTAINER"
  export DATASET_PATH="$DATASET_PATH_CONTAINER"
fi
{% endif %}
srun -A "$ACC" -p "{{ slurm_config.partition }}" -N "$N_NODES" -n1 -t "{{ slurm_config.time }}" --gpus-per-task="$GPUS" \
  singularity exec --rocm --cleanenv \
    --bind "$SCR":/workspace \
    --bind "$PRJ":/project \
    --bind /usr/share/libdrm:/usr/share/libdrm \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    --env MODEL_ID="$MODEL_ID" \
    --env SCR="$SCR" \
    --env CONTEXT_LENGTH={{ env_vars.CONTEXT_LENGTH }} \
    --env DATASET_SAMPLES={{ env_vars.DATASET_SAMPLES }} \
    --env ALPHA={{ env_vars.ALPHA }} \
    --env BETA={{ env_vars.BETA }} \
    {% if env_vars.LONGPPL_DATASET %}--env LONGPPL_DATASET="$DATASET_PATH" \
    {% endif %}--env HF_HOME=/project/hf_cache \
    --env HUGGINGFACE_HUB_CACHE=/project/hf_cache/hub \
    --env TRANSFORMERS_CACHE=/project/hf_cache/models \
    --env HF_DATASETS_CACHE=/project/hf_cache/datasets \
    --env XDG_CACHE_HOME=/project/hf_cache/xdg \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    "$IMG" bash -lc '
set -euo pipefail
umask 002

# ==================== STAGE SCRIPTS SETUP ====================

# Copy stage scripts from templates directory
cp /workspace/templates/longppl/setup.sh /workspace/tools/longppl_setup.sh
cp /workspace/templates/longppl/run_eval.sh /workspace/tools/longppl_run_eval.sh
cp /workspace/templates/longppl/parse_results.py /workspace/tools/longppl_parse_results.py

chmod +x /workspace/tools/longppl_setup.sh /workspace/tools/longppl_run_eval.sh

# ==================== EXECUTE STAGES ====================

echo "=========================================="
echo "LONGPPL EVALUATION - STAGED EXECUTION"
echo "=========================================="
echo "Model: $MODEL_ID"
echo "Context Length: $CONTEXT_LENGTH"
echo "Dataset Samples: $DATASET_SAMPLES"
echo "Alpha: $ALPHA, Beta: $BETA"
echo "=========================================="
echo

# Stage 1: Setup
echo "=========================================="
echo "STAGE 1: SETUP"
echo "=========================================="
bash /workspace/tools/longppl_setup.sh
STAGE1_EXIT=$?

if [ $STAGE1_EXIT -ne 0 ]; then
  echo "✗ Stage 1 (Setup) failed with exit code $STAGE1_EXIT"
  exit $STAGE1_EXIT
fi

echo
echo "=========================================="
echo "STAGE 2: RUN EVALUATION"
echo "=========================================="
bash /workspace/tools/longppl_run_eval.sh
STAGE2_EXIT=$?

if [ $STAGE2_EXIT -ne 0 ]; then
  echo "✗ Stage 2 (Run Evaluation) failed with exit code $STAGE2_EXIT"
  exit $STAGE2_EXIT
fi

echo
echo "=========================================="
echo "STAGE 3: PARSE RESULTS"
echo "=========================================="
# Source environment variables from setup stage
source /workspace/tools/longppl_env.sh
python /workspace/tools/longppl_parse_results.py \
  --log-file "$OUTPUT_DIR/longppl_${CONTEXT_LENGTH}.log" \
  --output-file "$OUTPUT_DIR/longppl_${CONTEXT_LENGTH}.json" \
  --model-id "$MODEL_ID" \
  --context-length $CONTEXT_LENGTH \
  --dataset-samples $DATASET_SAMPLES \
  --alpha $ALPHA \
  --beta $BETA \
  --slurm-job-id "$SLURM_JOB_ID"

STAGE3_EXIT=$?

if [ $STAGE3_EXIT -ne 0 ]; then
  echo "✗ Stage 3 (Parse Results) failed with exit code $STAGE3_EXIT"
  exit $STAGE3_EXIT
fi

echo
echo "=========================================="
echo "LONGPPL EVALUATION COMPLETE"
echo "=========================================="
echo "✓ All stages completed successfully"
echo
echo "Output files:"
ls -lh "$OUTPUT_DIR"/longppl_* || true
'
