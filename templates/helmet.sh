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
export PRJ="/scratch/{{ slurm_config.account }}"   # will be /project in container
export SCR="$PWD"                          # SCR = scratch directory, will be /workspace in container
export ACC="{{ slurm_config.account }}"

# Parse gres for GPU count (e.g., "gpu:mi250:4" -> 4)
GRES="{{ slurm_config.gres }}"
if [[ "$GRES" =~ gpu:[^:]*:([0-9]+) ]]; then
    GPUS="${BASH_REMATCH[1]}"
elif [[ "$GRES" =~ gpu:([0-9]+) ]]; then
    GPUS="${BASH_REMATCH[1]}"
else
    echo "Warning: Could not parse GPU count from GRES '$GRES', defaulting to 4"
    GPUS=4
fi

# topology & model knobs
export N_NODES=1
export TP="$GPUS"
export MODEL_ID="{{ env_vars.MODEL }}"

srun -A "$ACC" -p "{{ slurm_config.partition }}" -N "$N_NODES" -n1 -t "{{ slurm_config.time }}" --gpus-per-task="$GPUS" \
  singularity exec --rocm --cleanenv \
    --bind "$SCR":/workspace \
    --bind "$PRJ":/project \
    --bind /usr/share/libdrm:/usr/share/libdrm \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    --env MODEL_ID="$MODEL_ID" \
    --env TP="$TP" \
    --env SCR="$SCR" \
    --env HF_HOME=/project/hf_cache \
    --env HUGGINGFACE_HUB_CACHE=/project/hf_cache/hub \
    --env TRANSFORMERS_CACHE=/project/hf_cache/models \
    --env HF_DATASETS_CACHE=/project/hf_cache/datasets \
    --env XDG_CACHE_HOME=/project/hf_cache/xdg \
    "$IMG" bash -lc '
set -euo pipefail
umask 002

# ---- model/topology (now variables) ----
MODEL_ID="${MODEL_ID:?MODEL_ID not set}"
TP="${TP:-4}"
MODEL_SAFE="${MODEL_ID//\//-}"
PREFETCH_LOCAL_DIR="/project/hf_cache/models/${MODEL_SAFE}"
export MODEL_LOCAL="${PREFETCH_LOCAL_DIR}"

# ---- env & caches (disable xet + telemetry) ----
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1

export PATH="$HOME/.local/bin:/opt/miniconda3/envs/pytorch/bin:/opt/rocm/llvm/bin:/opt/rocm/bin:/usr/bin:/bin"
export HF_HOME=/project/hf_cache
export HUGGINGFACE_HUB_CACHE=/project/hf_cache/hub
export TRANSFORMERS_CACHE=/project/hf_cache/models
export HF_DATASETS_CACHE=/project/hf_cache/datasets
export XDG_CACHE_HOME=/project/hf_cache/xdg

# Debug: Print cache configuration
echo "DEBUG: Cache configuration:"
echo "  HF_HOME=${HF_HOME:-NOT_SET}"
echo "  HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-NOT_SET}"
echo "  HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-NOT_SET}"
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext
export TORCHINDUCTOR_CACHE_DIR=/project/hf_cache/torchinductor
export VLLM_COMPILER_CACHE_DIR=/project/hf_cache/vllm-compile

export VLLM_USE_V1=1
export VLLM_TARGET_DEVICE=rocm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HIP_ARCHITECTURES=gfx90a

# Avoid the 1-GPU trap - ensure PyTorch uses all allocated GPUs
unset HIP_VISIBLE_DEVICES

mkdir -p /project/hf_cache/{hub,models,datasets,torchinductor,xdg,vllm-compile} \
         /dev/shm/torch_ext "$HOME/.aiter/jit/build" "$HOME/.aiter/jit/install" \
         /workspace/tools

# Prefer ROCm clang toolchain (for Triton/Inductor & aiter)
if command -v /opt/rocm/llvm/bin/clang++ >/dev/null 2>&1; then
  export CC=/opt/rocm/llvm/bin/clang
  export CXX=/opt/rocm/llvm/bin/clang++
else
  export CC=/opt/rocm/bin/hipcc
  export CXX=/opt/rocm/bin/hipcc
fi

# Make sure ninja is available
python -m pip -q install --user -U ninja || true

# ------- write helper: sanity.py -------
cat > /workspace/tools/sanity.py <<PY
import torch, os
print("torch", torch.__version__, "HIP", getattr(torch.version, "hip", None))
print("HF_HUB_DISABLE_XET =", os.getenv("HF_HUB_DISABLE_XET"))
n = torch.cuda.device_count()
print("cuda.device_count =", n)
for i in range(n):
    print("  idx", i, "->", torch.cuda.get_device_name(i))
PY

python /workspace/tools/sanity.py

{% if env_vars.BACKEND != "dummy" %}
# ------- write helper: prefetch.py -------
cat > /workspace/tools/prefetch.py <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(
  repo_id="${MODEL_ID}",
  local_dir="${PREFETCH_LOCAL_DIR}",
  local_dir_use_symlinks=False,
  allow_patterns=["*.safetensors","*.json","tokenizer.*","*vocab*","*.model"]
)
print("prefetch OK ->", p)
PY

python /workspace/tools/prefetch.py
{% endif %}

# ------- Setup HELMET -------
HELMET_DIR="/workspace/helmet-${SLURM_JOB_ID:-$}"
echo "Setting up HELMET in $HELMET_DIR"

# Copy HELMET from workspace
cp -r /workspace/helmet "$HELMET_DIR"
cd "$HELMET_DIR"

# Install HELMET-specific dependencies (container already has torch, transformers, datasets, etc.)
echo "Installing HELMET-specific dependencies..."
# Create a temporary writable location for pip
PIP_TMP_DIR="/tmp/pip_install_${SLURM_JOB_ID:-$$}"
mkdir -p "$PIP_TMP_DIR"
export TMPDIR="$PIP_TMP_DIR/tmp"
export PIP_CACHE_DIR="$PIP_TMP_DIR/cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# Install to a writable location (not --user which tries to write to /pfs)
# Use --target to install to a specific directory
PIP_INSTALL_DIR="$PIP_TMP_DIR/packages"
mkdir -p "$PIP_INSTALL_DIR"
python -m pip install --target "$PIP_INSTALL_DIR" pytrec_eval rouge_score openai

# Add to Python path
export PYTHONPATH="$PIP_INSTALL_DIR:${PYTHONPATH:-}"

# Convert host paths to container paths for output file
CONTAINER_OUTPUT_FILE="{{ env_vars.OUTPUT_FILE }}"
if [[ -z "$CONTAINER_OUTPUT_FILE" ]]; then
  echo "ERROR: OUTPUT_FILE is empty. Check main.py env_vars." >&2
  exit 2
fi
USER_SCRATCH_DIR="/scratch/project_462000353/$USER"
PFS_USER_PREFIX="/pfs/lustrep2/scratch/project_462000353/$USER/"

echo "DEBUG: Original OUTPUT_FILE: $CONTAINER_OUTPUT_FILE"
echo "DEBUG: SCR inside container: $SCR"

# Convert various host path patterns to container paths
if [[ "$CONTAINER_OUTPUT_FILE" == "$SCR"* ]]; then
    echo "DEBUG: Path matches SCR prefix, converting..."
    RELATIVE_PATH="${CONTAINER_OUTPUT_FILE#$SCR}"
    RELATIVE_PATH="${RELATIVE_PATH#/}"
    CONTAINER_OUTPUT_FILE="/workspace/${RELATIVE_PATH}"
    echo "DEBUG: Converted to: $CONTAINER_OUTPUT_FILE"
elif [[ "$CONTAINER_OUTPUT_FILE" == "$PFS_USER_PREFIX"* ]]; then
    echo "DEBUG: Path matches PFS prefix, converting..."
    RELATIVE_PATH="${CONTAINER_OUTPUT_FILE#$PFS_USER_PREFIX}"
    RELATIVE_PATH="${RELATIVE_PATH#evals/}"
    CONTAINER_OUTPUT_FILE="/workspace/${RELATIVE_PATH}"
    echo "DEBUG: Converted to: $CONTAINER_OUTPUT_FILE"
elif [[ "$CONTAINER_OUTPUT_FILE" == "$USER_SCRATCH_DIR"* ]]; then
    echo "DEBUG: Path matches USER_SCRATCH_DIR prefix, converting..."
    RELATIVE_PATH="${CONTAINER_OUTPUT_FILE#$USER_SCRATCH_DIR}"
    RELATIVE_PATH="${RELATIVE_PATH#/}"
    CONTAINER_OUTPUT_FILE="/workspace/${RELATIVE_PATH}"
    echo "DEBUG: Converted to: $CONTAINER_OUTPUT_FILE"
else
    echo "DEBUG: No path conversion matched - keeping original path"
fi

# Prepare final output directory inside container
HELMET_OUTPUT_DIR="$(dirname "$CONTAINER_OUTPUT_FILE")"
mkdir -p "$HELMET_OUTPUT_DIR"
echo "HELMET results will be saved to: $HELMET_OUTPUT_DIR"

# Set up backend-specific arguments
{% if env_vars.BACKEND == "vllm" %}
# VLLM backend - use HELMET's --use_vllm flag
BACKEND_ARGS="--use_vllm"
echo "Using VLLM backend"
{% elif env_vars.BACKEND == "dummy" %}
# Dummy backend - skip evaluation
echo "Dummy backend selected - skipping actual evaluation"
exit 0
{% else %}
# HuggingFace backend configuration (default)
BACKEND_ARGS=""
echo "Using HuggingFace backend"
{% endif %}

# Set up chat template flags
{% if env_vars.APPLY_CHAT_TEMPLATE %}
CHAT_TEMPLATE_FLAG="--use_chat_template"
{% else %}
CHAT_TEMPLATE_FLAG=""
{% endif %}

# ------- Run HELMET evaluation -------
echo "Running HELMET evaluation with config: {{ env_vars.CONFIG_NAME }}"
echo "DEBUG: MODEL_LOCAL=${MODEL_LOCAL}"
echo "DEBUG: HELMET_OUTPUT_DIR=${HELMET_OUTPUT_DIR}"
python eval.py \
  --config configs/{{ env_vars.CONFIG_NAME }}.yaml \
  --model_name_or_path "${MODEL_LOCAL}" \
  --output_dir "$HELMET_OUTPUT_DIR" \
  $BACKEND_ARGS \
  $CHAT_TEMPLATE_FLAG \
  --overwrite

# Move results to final output file
echo "Moving HELMET results to $CONTAINER_OUTPUT_FILE"
# HELMET creates multiple output files, we need to collect them
find "$HELMET_OUTPUT_DIR" -name "*.json" -o -name "*.json.score" | while read -r result_file; do
    basename=$(basename "$result_file")
    if [[ ! -f "${HELMET_OUTPUT_DIR}/${basename}" ]]; then
        cp "$result_file" "${HELMET_OUTPUT_DIR}/"
    fi
done

# Create a summary file at the expected output location
cat > "$CONTAINER_OUTPUT_FILE" <<EOF
{
  "helmet_config": "{{ env_vars.CONFIG_NAME }}",
  "model": "${MODEL_ID}",
  "output_dir": "$HELMET_OUTPUT_DIR",
  "results": "See individual result files in output directory"
}
EOF

echo "== HELMET evaluation complete =="
echo "Results saved to: $HELMET_OUTPUT_DIR"
ls -lh "$HELMET_OUTPUT_DIR"

# Clean up the temporary HELMET directory
echo "Cleaning up temporary HELMET directory: $HELMET_DIR"
cd /workspace
rm -rf "$HELMET_DIR"
'
