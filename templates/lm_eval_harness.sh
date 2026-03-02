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

ln -sf {{ slurm_config.log_dir }}/$SLURM_JOB_ID.out {{ slurm_config.log_dir }}/latest.out
ln -sf {{ slurm_config.log_dir }}/$SLURM_JOB_ID.err {{ slurm_config.log_dir }}/latest.err

set -euo pipefail

echo "Starting lm_eval job..."
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# LUMI AI Factory container (Ubuntu 24.04, ROCm 6.4, vLLM 0.12)
export IMG="/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20251209_134408/lumi-multitorch-full-u24r64f21m43t29-20251209_134408.sif"
# Storage project (use job's account project for storage)
export STORAGE_PRJ="/scratch/{{ slurm_config.account }}"
export SCR="$(pwd -P)"
export ACC="{{ slurm_config.account }}"

# Parse gres for GPU count
GRES="{{ slurm_config.gres }}"
if [[ "$GRES" =~ gpu:[^:]*:([0-9]+) ]]; then
    GPUS="${BASH_REMATCH[1]}"
elif [[ "$GRES" =~ gpu:([0-9]+) ]]; then
    GPUS="${BASH_REMATCH[1]}"
else
    GPUS=1
fi

export MODEL="{{ env_vars.MODEL }}"
export TP="$GPUS"

echo "Container: $IMG"
echo "Model: $MODEL"
echo "GPUs: $GPUS"
echo "Storage: $STORAGE_PRJ"

{% if env_vars.LM_EVAL_PATH %}
# Bind mount local lm-eval path into container
BIND_LM_EVAL="--bind {{ env_vars.LM_EVAL_PATH }}:/workspace/lm-eval-host"
{% else %}
BIND_LM_EVAL=""
{% endif %}

# Use srun to properly allocate GPUs to the container
srun -A "$ACC" -p "{{ slurm_config.partition }}" -N1 -n1 -t "{{ slurm_config.time }}" --gpus-per-task="$GPUS" \
  singularity exec --rocm --cleanenv \
    --bind "$SCR":/workspace \
    --bind "$STORAGE_PRJ":/project \
    --bind /pfs,/scratch,/projappl,/flash,/appl \
    --bind /var/spool/slurmd \
    --bind /opt/cray/ \
    --bind /usr/lib64/libcxi.so.1 \
    --bind /usr/share/libdrm:/usr/share/libdrm \
    $BIND_LM_EVAL \
    --env MODEL="$MODEL" \
    --env TP="$TP" \
    --env SCR="$SCR" \
    --env USER="$USER" \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    --env HF_HOME=/project/cache/huggingface \
    --env HUGGINGFACE_HUB_CACHE=/project/cache/huggingface/hub \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env STORAGE_PRJ="$STORAGE_PRJ" \
    "$IMG" bash -c '
set -euo pipefail
echo "Inside container..."

# Set HOME to /tmp for ephemeral builds
export HOME=/tmp

source /opt/venv/bin/activate
PYTHON_BIN="/opt/venv/bin/python"

echo "Python: $PYTHON_BIN"
echo "ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-not set}"
echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-not set}"
$PYTHON_BIN -c "import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}\")"
$PYTHON_BIN -c "import vllm; print(f\"vLLM: {vllm.__version__}\")"

export HF_HUB_DISABLE_XET=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export PYTORCH_ROCM_ARCH=gfx90a

export VLLM_USE_V1=1
export VLLM_TARGET_DEVICE=rocm
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p /project/cache/huggingface/hub "$MIOPEN_USER_DB_PATH"

# Clean up any leftover files from previous runs
rm -rf /tmp/pip-packages /tmp/lm-eval

# Install transformers
PIP_TARGET=/tmp/pip-packages
mkdir -p "$PIP_TARGET"
echo "Installing transformers {{ env_vars.TRANSFORMERS_VERSION }}..."
$PYTHON_BIN -m pip install -q --target="$PIP_TARGET" "numpy<2.3" "transformers=={{ env_vars.TRANSFORMERS_VERSION }}" "hf_transfer"
export PYTHONPATH="$PIP_TARGET:${PYTHONPATH:-}"

# Install RULER dependencies if running RULER tasks
{% if env_vars.MAX_SEQ_LENGTH %}
echo "Installing RULER dependencies (wonderwords, nltk)..."
$PYTHON_BIN -m pip install -q --target="$PIP_TARGET" wonderwords nltk
echo "RULER dependencies installed successfully"
{% endif %}

EVAL_HARNESS_DIR="/tmp/lm-eval"
echo "Setting up lm-evaluation-harness in $EVAL_HARNESS_DIR"

{% if env_vars.LM_EVAL_PATH %}
echo "Using local lm-evaluation-harness from: {{ env_vars.LM_EVAL_PATH }}"
cp -r "/workspace/lm-eval-host" "$EVAL_HARNESS_DIR"
{% else %}
REPO_URL="{{ env_vars.LM_EVAL_REPO }}"
REPO_REF="{{ env_vars.LM_EVAL_REF }}"
echo "Cloning lm-evaluation-harness from $REPO_URL (ref: $REPO_REF)..."
git clone --depth 1 -b "$REPO_REF" "$REPO_URL" "$EVAL_HARNESS_DIR"
{% endif %}

export PYTHONPATH="$EVAL_HARNESS_DIR:${PYTHONPATH:-}"

# Remap output file path from host to container if needed
OUTPUT_FILE="{{ env_vars.OUTPUT_FILE }}"
if [[ "$OUTPUT_FILE" == "$SCR"* ]]; then
  OUTPUT_FILE="/workspace${OUTPUT_FILE#$SCR}"
elif [[ "$OUTPUT_FILE" != /* ]]; then
  OUTPUT_FILE="/workspace/$OUTPUT_FILE"
fi
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Remap MODEL from host paths to container paths if needed
if [[ "$MODEL" == "$STORAGE_PRJ"/* ]]; then
  MODEL="/project${MODEL#$STORAGE_PRJ}"
elif [[ "$MODEL" == "$SCR"/* ]]; then
  MODEL="/workspace${MODEL#$SCR}"
fi

{% if env_vars.APPLY_CHAT_TEMPLATE %}
CHAT_TEMPLATE_FLAG="--apply_chat_template"
{% else %}
CHAT_TEMPLATE_FLAG=""
{% endif %}

{% if env_vars.FEWSHOT_AS_MULTITURN %}
FEWSHOT_AS_MULTITURN_FLAG="--fewshot_as_multiturn"
{% else %}
FEWSHOT_AS_MULTITURN_FLAG=""
{% endif %}

{% if env_vars.BACKEND == "vllm" %}
MODEL_BACKEND="vllm"
BASE_ARGS="pretrained=${MODEL},dtype=auto,tensor_parallel_size=${TP}"

{% if env_vars.MAX_SEQ_LENGTH %}
# RULER task: Force max_model_len to match RULER sequence length
# Remove any max_model_len setting from MODEL_ARGS if present, then add RULER length
{% if env_vars.MODEL_ARGS %}
EXTRA_ARGS="{{ env_vars.MODEL_ARGS }}"
# Remove any max_model_len setting from extra args
EXTRA_ARGS=$(echo "$EXTRA_ARGS" | sed "s/max_model_len=[0-9]*,\?//g" | sed "s/,,/,/g" | sed "s/^,//;s/,\$//")
{% else %}
EXTRA_ARGS=""
{% endif %}
# Add RULER max_model_len (must match sequence length for RULER)
if [ -n "$EXTRA_ARGS" ]; then
    MODEL_ARGS="${BASE_ARGS},${EXTRA_ARGS},max_model_len={{ env_vars.MAX_SEQ_LENGTH }},gpu_memory_utilization=0.90"
else
    MODEL_ARGS="${BASE_ARGS},max_model_len={{ env_vars.MAX_SEQ_LENGTH }},gpu_memory_utilization=0.90"
fi
echo "RULER: Using max_model_len={{ env_vars.MAX_SEQ_LENGTH }} (matching RULER sequence length)"
{% else %}
# Non-RULER task: Use default or provided max_model_len
MODEL_ARGS="${BASE_ARGS},gpu_memory_utilization=0.90"
{% if env_vars.MODEL_ARGS %}
MODEL_ARGS="${MODEL_ARGS},{{ env_vars.MODEL_ARGS }}"
{% endif %}
{% endif %}
{% elif env_vars.BACKEND == "dummy" %}
MODEL_BACKEND="dummy"
MODEL_ARGS="pretrained=${MODEL}"
{% else %}
MODEL_BACKEND="hf-auto"
BASE_ARGS="pretrained=${MODEL},device_map=auto,dtype=bfloat16,trust_remote_code=True,attn_implementation=sdpa"

{% if env_vars.MAX_SEQ_LENGTH %}
# RULER task: Force max_length to match RULER sequence length
# Strip any max_length from MODEL_ARGS if present, then add RULER length
{% if env_vars.MODEL_ARGS %}
EXTRA_ARGS="{{ env_vars.MODEL_ARGS }}"
# Remove any max_length setting from extra args
EXTRA_ARGS=$(echo "$EXTRA_ARGS" | sed "s/max_length=[0-9]*,\?//g" | sed "s/,,/,/g" | sed "s/^,//;s/,\$//")
{% else %}
EXTRA_ARGS=""
{% endif %}
if [ -n "$EXTRA_ARGS" ]; then
    MODEL_ARGS="${BASE_ARGS},${EXTRA_ARGS},max_length={{ env_vars.MAX_SEQ_LENGTH }}"
else
    MODEL_ARGS="${BASE_ARGS},max_length={{ env_vars.MAX_SEQ_LENGTH }}"
fi
echo "RULER: Using max_length={{ env_vars.MAX_SEQ_LENGTH }} (matching RULER sequence length)"
{% else %}
# Non-RULER task: Use provided or default settings
{% if env_vars.MODEL_ARGS %}
MODEL_ARGS="${BASE_ARGS},{{ env_vars.MODEL_ARGS }}"
{% else %}
MODEL_ARGS="${BASE_ARGS}"
{% endif %}
{% endif %}
{% endif %}

BATCH_SIZE="{% if env_vars.BATCH_SIZE %}{{ env_vars.BATCH_SIZE }}{% elif env_vars.BACKEND == "vllm" %}auto{% else %}4{% endif %}"

{% if env_vars.MAX_SEQ_LENGTH %}
# Set up metadata for RULER tasks
TASK_METADATA_FLAG="--metadata {\"max_seq_lengths\":[{{ env_vars.MAX_SEQ_LENGTH }}]}"
echo "Using RULER metadata for sequence length: {{ env_vars.MAX_SEQ_LENGTH }}"
{% else %}
TASK_METADATA_FLAG=""
{% endif %}

echo "Running lm_eval with model: $MODEL"
$PYTHON_BIN -m lm_eval \
  --model "$MODEL_BACKEND" \
  --model_args "$MODEL_ARGS" \
  --tasks "{{ env_vars.TASK_LIST }}" \
  --num_fewshot {{ env_vars.NUM_FEWSHOT }} \
  --batch_size "$BATCH_SIZE" \
  --output_path "$OUTPUT_FILE" \
  $CHAT_TEMPLATE_FLAG \
  $FEWSHOT_AS_MULTITURN_FLAG \
  $TASK_METADATA_FLAG \
{% if env_vars.LIMIT %}  --limit {{ env_vars.LIMIT }} \
{% endif %}  --log_samples \
{% if env_vars.LM_EVAL_ARGS %}  {{ env_vars.LM_EVAL_ARGS }}
{% endif %}

echo "lm_eval completed!"
'

echo "Job finished"
