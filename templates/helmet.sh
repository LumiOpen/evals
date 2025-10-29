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
    --env HF_TOKEN="${HF_TOKEN:-}" \
    "$IMG" bash -lc '
set -euo pipefail
umask 002

# ---- model/topology (now variables) ----
export MODEL_ID="${MODEL_ID:?MODEL_ID not set}"
export TP="${TP:-4}"
export MODEL_SAFE="${MODEL_ID//\//-}"
export PREFETCH_LOCAL_DIR="/project/hf_cache/models/${MODEL_SAFE}"
export MODEL_LOCAL="${PREFETCH_LOCAL_DIR}"
echo "DEBUG: Set MODEL_LOCAL=$MODEL_LOCAL"

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
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

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

# ------- write helper: stage_aiter.py -------
cat > /workspace/tools/stage_aiter.py <<PY
import os, glob, shutil, importlib, pathlib, subprocess, sys

home = os.path.expanduser("~")
jit_root   = os.path.join(home, ".aiter", "jit")
build_root = os.path.join(jit_root, "build")
inst_root  = os.path.join(jit_root, "install")
pkg_root   = os.path.join(inst_root, "private_aiter")
pkg_jit    = os.path.join(pkg_root, "jit")

os.makedirs(pkg_jit, exist_ok=True)
pathlib.Path(os.path.join(pkg_root, "__init__.py")).write_text("")
pathlib.Path(os.path.join(pkg_jit, "__init__.py")).write_text("")

# trigger a build once (ok if it raises)
try:
    import aiter
    from aiter.ops import enum  # will build module_aiter_enum
except Exception as e:
    print("[aiter] prewarm raised:", repr(e))

hits = glob.glob(os.path.join(build_root, "**", "module_aiter_enum*.so"), recursive=True)
if not hits:
    raise SystemExit("[stage] no compiled module_aiter_enum*.so found under " + build_root)

so_src = max(hits, key=os.path.getmtime)
dst = os.path.join(pkg_jit, "module_aiter_enum.so")
if os.path.islink(dst) or os.path.exists(dst):
    os.remove(dst)
try:
    os.symlink(so_src, dst)
    print("[stage] symlinked", dst, "->", so_src)
except OSError:
    shutil.copy2(so_src, dst)
    print("[stage] copied", so_src, "->", dst)

print("[ldd]")
print(subprocess.check_output(["ldd", dst], text=True))

sys.path.insert(0, inst_root)
m = importlib.import_module("private_aiter.jit.module_aiter_enum")
print("[stage] import OK:", m.__spec__.origin)

import aiter; from aiter.ops import enum as _e
print("[stage] aiter import OK")
PY

python /workspace/tools/stage_aiter.py

{% if env_vars.BACKEND != "dummy" %}
# ------- write helper: prefetch.py -------
# Only prefetch if MODEL_ID is not a local path
if [[ "${MODEL_ID}" == /* ]]; then
  echo "Model is a local path, skipping prefetch"
  # For local paths, use the path directly
  export MODEL_LOCAL="${MODEL_ID}"
else
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
fi
{% endif %}

# ------- Setup HELMET -------
# Use HELMET directly from /workspace to allow data caching
export HELMET_DIR="/workspace/helmet"
echo "Using HELMET from $HELMET_DIR"
cd "$HELMET_DIR"
echo "DEBUG: After cd to HELMET_DIR, MODEL_LOCAL=$MODEL_LOCAL"

# Download HELMET data if not already present
if [ ! -d "data" ]; then
    echo "Downloading HELMET benchmark data..."
    curl -L -o data.tar.gz https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/main/data.tar.gz
    tar -xzf data.tar.gz
    rm data.tar.gz
    echo "HELMET data downloaded and extracted"
else
    echo "HELMET data already present, skipping download"
fi

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
# Use --no-deps to avoid installing incompatible numpy 2.3 (container has compatible versions)
PIP_INSTALL_DIR="$PIP_TMP_DIR/packages"
mkdir -p "$PIP_INSTALL_DIR"
python -m pip install --target "$PIP_INSTALL_DIR" --no-deps pytrec_eval rouge_score openai safetensors
# Install only the openai dependencies not already in container
python -m pip install --target "$PIP_INSTALL_DIR" --no-deps anyio distro httpx jiter pydantic sniffio tqdm typing-extensions certifi httpcore h11 idna annotated-types pydantic-core typing-inspection

# Download NLTK data for ALCE citation scoring
/opt/miniconda3/envs/pytorch/bin/python -m nltk.downloader punkt_tab -d /users/danizaut/nltk_data || echo "Warning: Could not download punkt_tab"

# Add to Python path (including aiter install location)
export PYTHONPATH="$HOME/.aiter/jit/install:$PIP_INSTALL_DIR:${PYTHONPATH:-}"

# Hardcode output paths - /workspace is bound to the evals repo dir
# Output goes to /workspace/output/v2/model_org/model_name/
# Handle both HuggingFace repo IDs and local paths
if [[ "${MODEL_ID}" == /* ]]; then
  # Local path - just use basename for now
  MODEL_BASENAME=$(basename "$MODEL_ID")
  export HELMET_OUTPUT_DIR="/workspace/output/v2/local/${MODEL_BASENAME}"
else
  # HuggingFace repo ID - use org/name structure
  export MODEL_ORG=$(echo "$MODEL_ID" | cut -d/ -f1)
  export MODEL_NAME=$(echo "$MODEL_ID" | cut -d/ -f2)
  export HELMET_OUTPUT_DIR="/workspace/output/v2/${MODEL_ORG}/${MODEL_NAME}"
fi
mkdir -p "$HELMET_OUTPUT_DIR"
echo "HELMET results will be saved to: $HELMET_OUTPUT_DIR"

# Set up backend-specific arguments
{% if env_vars.BACKEND == "vllm" %}
# VLLM backend - use HELMET --use_vllm flag
export BACKEND_ARGS="--use_vllm"
echo "Using VLLM backend"
{% elif env_vars.BACKEND == "dummy" %}
# Dummy backend - skip evaluation
echo "Dummy backend selected - skipping actual evaluation"
exit 0
{% else %}
# HuggingFace backend configuration (default)
export BACKEND_ARGS=""
echo "Using HuggingFace backend"
{% endif %}

# Set up chat template flags
{% if env_vars.APPLY_CHAT_TEMPLATE %}
export CHAT_TEMPLATE_FLAG="--use_chat_template True"
{% else %}
export CHAT_TEMPLATE_FLAG="--use_chat_template False"
{% endif %}

# Set up RoPE scaling flags
{% if env_vars.ROPE_SCALING_FACTOR and env_vars.ROPE_SCALING_TYPE %}
export ROPE_FLAGS="--rope_scaling_factor {{ env_vars.ROPE_SCALING_FACTOR }} --rope_scaling_type {{ env_vars.ROPE_SCALING_TYPE }}"
echo "Using RoPE scaling: factor={{ env_vars.ROPE_SCALING_FACTOR }}, type={{ env_vars.ROPE_SCALING_TYPE }}"
{% else %}
export ROPE_FLAGS=""
{% endif %}

# ------- Run HELMET evaluation -------
echo "Running HELMET evaluation with config: {{ env_vars.CONFIG_NAME }}"
echo "DEBUG: MODEL_ID=$MODEL_ID"
echo "DEBUG: About to check HELMET_OUTPUT_DIR..."
set | grep HELMET_OUTPUT_DIR || echo "HELMET_OUTPUT_DIR not in environment!"
echo "DEBUG: HELMET_OUTPUT_DIR=${HELMET_OUTPUT_DIR}"
echo "DEBUG: BACKEND_ARGS=${BACKEND_ARGS}"
echo "DEBUG: CHAT_TEMPLATE_FLAG=${CHAT_TEMPLATE_FLAG}"

python eval.py \
  --config configs/{{ env_vars.CONFIG_NAME }}.yaml \
  --model_name_or_path "$MODEL_ID" \
  --output_dir "$HELMET_OUTPUT_DIR" \
  $BACKEND_ARGS \
  $CHAT_TEMPLATE_FLAG \
  $ROPE_FLAGS \
  --overwrite

echo "== HELMET evaluation complete =="
echo "Results saved to: $HELMET_OUTPUT_DIR"
ls -lh "$HELMET_OUTPUT_DIR" || true
'
