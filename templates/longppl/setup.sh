#!/bin/bash
# LongPPL Setup Stage
# Sets up environment, prefetches model, installs dependencies, and prepares output directory

set -euo pipefail

echo "=== LongPPL Setup Stage ==="

# ---- Model setup ----
export MODEL_ID="${MODEL_ID:?MODEL_ID not set}"
export MODEL_SAFE="${MODEL_ID//\\//-}"

# ---- Environment & caches ----
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
python -c 'import torch; print(f"PyTorch: {torch.__version__}, CUDA devices: {torch.cuda.device_count()}")'

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

# Install LongPPL dependencies (lightweight, only what container doesn't have)
echo "Installing LongPPL dependencies..."
PIP_TMP_DIR="/tmp/pip_install_${SLURM_JOB_ID:-$$}"
mkdir -p "$PIP_TMP_DIR"
export TMPDIR="$PIP_TMP_DIR/tmp"
export PIP_CACHE_DIR="$PIP_TMP_DIR/cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

PIP_INSTALL_DIR="$PIP_TMP_DIR/packages"
mkdir -p "$PIP_INSTALL_DIR"

# Install only missing packages (skip pytrec_eval - needs gcc, not used by LongPPL)
# Note: torch, transformers, tqdm, numpy should already be in container
# Install datasets without --no-deps to get necessary dependencies
python -m pip install --target "$PIP_INSTALL_DIR" datasets evaluate rouge_score || true
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

# Write environment variables for subsequent stages
cat > /workspace/tools/longppl_env.sh <<EOF
export MODEL_LOCAL="$MODEL_LOCAL"
export OUTPUT_DIR="$OUTPUT_DIR"
export CONTEXT_LENGTH="$CONTEXT_LENGTH"
export DATASET_SAMPLES="$DATASET_SAMPLES"
export ALPHA="$ALPHA"
export BETA="$BETA"
EOF

echo "✓ Setup complete"
echo "  MODEL_LOCAL=$MODEL_LOCAL"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
