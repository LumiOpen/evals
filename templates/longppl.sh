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

srun -A "$ACC" -p "{{ slurm_config.partition }}" -N "$N_NODES" -n1 -t "{{ slurm_config.time }}" --gpus-per-task="$GPUS" \
  singularity exec --rocm --cleanenv \
    --bind "$SCR":/workspace \
    --bind "$PRJ":/project \
    --bind /usr/share/libdrm:/usr/share/libdrm \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    --env MODEL_ID="$MODEL_ID" \
    --env SCR="$SCR" \
    --env HF_HOME=/project/hf_cache \
    --env HUGGINGFACE_HUB_CACHE=/project/hf_cache/hub \
    --env TRANSFORMERS_CACHE=/project/hf_cache/models \
    --env HF_DATASETS_CACHE=/project/hf_cache/datasets \
    --env XDG_CACHE_HOME=/project/hf_cache/xdg \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    "$IMG" bash -lc '
set -euo pipefail
umask 002

# ---- model setup ----
export MODEL_ID="${MODEL_ID:?MODEL_ID not set}"
export MODEL_SAFE="${MODEL_ID//\//-}"

# ---- env & caches ----
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1

export PATH="$HOME/.local/bin:/opt/miniconda3/envs/pytorch/bin:/opt/rocm/llvm/bin:/opt/rocm/bin:/usr/bin:/bin"
export HF_HOME=/project/hf_cache
export HUGGINGFACE_HUB_CACHE=/project/hf_cache/hub
export TRANSFORMERS_CACHE=/project/hf_cache/models
export HF_DATASETS_CACHE=/project/hf_cache/datasets
export XDG_CACHE_HOME=/project/hf_cache/xdg

mkdir -p /project/hf_cache/{hub,models,datasets,xdg}

echo "=== Environment Setup ==="
echo "MODEL_ID: $MODEL_ID"
python -c "import torch; print(f\"PyTorch: {torch.__version__}, CUDA devices: {torch.cuda.device_count()}\")"

# ---- Prefetch model if HuggingFace repo ID ----
if [[ "${MODEL_ID}" == /* ]]; then
  echo "Model is a local path, using directly: ${MODEL_ID}"
  export MODEL_LOCAL="${MODEL_ID}"
else
  echo "Prefetching HuggingFace model: ${MODEL_ID}"
  export PREFETCH_LOCAL_DIR="/project/hf_cache/models/${MODEL_SAFE}"
  export MODEL_LOCAL="${PREFETCH_LOCAL_DIR}"

  python -c "
from huggingface_hub import snapshot_download
p = snapshot_download(
  repo_id=\"${MODEL_ID}\",
  local_dir=\"${PREFETCH_LOCAL_DIR}\",
  local_dir_use_symlinks=False,
  allow_patterns=[\"*.safetensors\",\"*.json\",\"tokenizer.*\",\"*vocab*\",\"*.model\"]
)
print(f\"Model prefetched to: {p}\")
"
fi

echo "Using model from: $MODEL_LOCAL"

# ==================== LONGPPL SETUP ====================

# Setup LongPPL directory
export LONGPPL_DIR="/workspace/longppl"
echo "Using LongPPL from $LONGPPL_DIR"
cd "$LONGPPL_DIR"

# Install LongPPL dependencies (lightweight, only what container doesnt have)
echo "Installing LongPPL dependencies..."
PIP_TMP_DIR="/tmp/pip_install_${SLURM_JOB_ID:-$$}"
mkdir -p "$PIP_TMP_DIR"
export TMPDIR="$PIP_TMP_DIR/tmp"
export PIP_CACHE_DIR="$PIP_TMP_DIR/cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

PIP_INSTALL_DIR="$PIP_TMP_DIR/packages"
mkdir -p "$PIP_INSTALL_DIR"

# Install only missing packages (skip pytrec_eval - needs gcc, not used by LongPPL)
python -m pip install --target "$PIP_INSTALL_DIR" --no-deps evaluate rouge_score 2>&1 | grep -v "Requirement already satisfied" || true

# Add to Python path (parent of longppl directory so 'import longppl' works)
export PYTHONPATH="/workspace:$PIP_INSTALL_DIR:${PYTHONPATH:-}"

# Determine output directory
if [[ "${MODEL_ID}" == /* ]]; then
  MODEL_BASENAME=$(basename "$MODEL_ID")
  export OUTPUT_DIR="/workspace/output/v2/local/${MODEL_BASENAME}"
else
  export MODEL_ORG=$(echo "$MODEL_ID" | cut -d/ -f1)
  export MODEL_NAME=$(echo "$MODEL_ID" | cut -d/ -f2)
  export OUTPUT_DIR="/workspace/output/v2/${MODEL_ORG}/${MODEL_NAME}"
fi
mkdir -p "$OUTPUT_DIR"

# LongPPL evaluation parameters
CONTEXT_LENGTH={{ env_vars.CONTEXT_LENGTH }}
DATASET_SAMPLES={{ env_vars.DATASET_SAMPLES }}
ALPHA={{ env_vars.ALPHA }}
BETA={{ env_vars.BETA }}

echo "=== LongPPL Evaluation Configuration ==="
echo "Model: $MODEL_ID"
echo "Model path: $MODEL_LOCAL"
echo "Context length: $CONTEXT_LENGTH"
echo "Dataset samples: $DATASET_SAMPLES"
echo "Alpha threshold: $ALPHA"
echo "Beta threshold: $BETA"
echo "Output directory: $OUTPUT_DIR"
echo

# Run LongPPL evaluation
cd /workspace/longppl/perplexity

echo "=== Starting LongPPL evaluation ==="
python perplexity.py \
  --dataset emozilla/govreport-test-tokenized \
  --tokenized \
  --dataset-min-tokens "$CONTEXT_LENGTH" \
  --samples "$DATASET_SAMPLES" \
  --model "$MODEL_LOCAL" \
  --evaluator-model "$MODEL_LOCAL" \
  --mode online \
  --max-length "$CONTEXT_LENGTH" \
  --alpha "$ALPHA" \
  --beta "$BETA" \
  2>&1 | tee "$OUTPUT_DIR/longppl_${CONTEXT_LENGTH}.log"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
  echo "=== LongPPL evaluation completed successfully ==="

  # Parse output and save structured results
  echo "Parsing results and creating JSON output..."
  python -c "
import re, json, sys, os
from datetime import datetime

log_file = \"$OUTPUT_DIR/longppl_${CONTEXT_LENGTH}.log\"
if not os.path.exists(log_file):
    print(f\"Error: Log file not found: {log_file}\")
    sys.exit(1)

with open(log_file) as f:
    log_text = f.read()

# Extract longppl and ppl values from output
# Format: \"model_name: longppl: XX.XX, ppl: YY.YY\"
longppl_match = re.search(r\"longppl:\s*([\d.]+)\", log_text)
ppl_match = re.search(r\"ppl:\s*([\d.]+)\", log_text)

if not longppl_match or not ppl_match:
    print(\"Warning: Could not parse longppl/ppl from output\")
    print(\"Log excerpt:\")
    print(log_text[-500:])

results = {
    \"model\": \"$MODEL_ID\",
    \"context_length\": $CONTEXT_LENGTH,
    \"dataset\": \"govreport-test-tokenized\",
    \"dataset_samples\": $DATASET_SAMPLES,
    \"alpha\": $ALPHA,
    \"beta\": $BETA,
    \"longppl\": float(longppl_match.group(1)) if longppl_match else None,
    \"ppl\": float(ppl_match.group(1)) if ppl_match else None,
    \"timestamp\": datetime.now().isoformat(),
    \"slurm_job_id\": \"$SLURM_JOB_ID\"
}

output_file = \"$OUTPUT_DIR/longppl_${CONTEXT_LENGTH}.json\"
with open(output_file, \"w\") as f:
    json.dump(results, f, indent=2)

print(f\"\\n=== Results Summary ===\")
print(f\"LongPPL: {results[\\\"longppl\\\"]}\")
print(f\"Standard PPL: {results[\\\"ppl\\\"]}\")
print(f\"Results saved to: {output_file}\")
"

  echo
  echo "=== Output files ==="
  ls -lh "$OUTPUT_DIR"/longppl_* || true

else
  echo "=== LongPPL evaluation failed with exit code $EXIT_CODE ==="
  exit $EXIT_CODE
fi
'
