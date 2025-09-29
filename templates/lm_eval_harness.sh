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

{% if env_vars.LM_EVAL_PATH %}
# Bind mount local lm-eval path into container
BIND_LM_EVAL="--bind {{ env_vars.LM_EVAL_PATH }}:/workspace/lm-eval-host"
{% else %}
BIND_LM_EVAL=""
{% endif %}

srun -A "$ACC" -p "{{ slurm_config.partition }}" -N "$N_NODES" -n1 -t "{{ slurm_config.time }}" --gpus-per-task="$GPUS" \
  singularity exec --rocm --cleanenv \
    --bind "$SCR":/workspace \
    --bind "$PRJ":/project \
    --bind /usr/share/libdrm:/usr/share/libdrm \
    $BIND_LM_EVAL \
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
MODEL_LOCAL="${PREFETCH_LOCAL_DIR}"

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

# ------- write helper: stage_aiter.py (NO stdin execution) -------
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

# ------- run the helpers from files (no <stdin>) -------
python /workspace/tools/sanity.py
python /workspace/tools/stage_aiter.py
python /workspace/tools/prefetch.py

# Model is prefetched, but allow datasets to be downloaded as needed
# Note: Datasets will be cached automatically for subsequent runs

# ensure staged package is visible
export PYTHONPATH="$HOME/.aiter/jit/install:${PYTHONPATH-}"

# ------- get LUMI harness (puts it first on sys.path) -------
# Use job-specific directory to avoid git conflicts between parallel jobs
EVAL_HARNESS_DIR="/workspace/lm-eval-${SLURM_JOB_ID:-$}"

# Function to setup lm-evaluation-harness (no locking needed with job-specific dirs)
setup_lm_eval() {
    echo "Setting up lm-evaluation-harness in $EVAL_HARNESS_DIR"

{% if env_vars.LM_EVAL_PATH %}
    # Use local path - copy from bind-mounted location
    echo "Using local lm-evaluation-harness from: {{ env_vars.LM_EVAL_PATH }}"
    cp -r "/workspace/lm-eval-host" "$EVAL_HARNESS_DIR"
{% else %}
    # Use git repository
    REPO_URL="{{ env_vars.LM_EVAL_REPO }}"
    REPO_REF="{{ env_vars.LM_EVAL_REF }}"

    echo "Cloning lm-evaluation-harness from $REPO_URL (ref: $REPO_REF)..."
    git clone --depth 1 -b "$REPO_REF" "$REPO_URL" "$EVAL_HARNESS_DIR"
{% endif %}
}

# Setup lm-evaluation-harness
setup_lm_eval
export PYTHONPATH="$EVAL_HARNESS_DIR:$PYTHONPATH"

# Create a temporary directory for lm_eval output (like HF template)
RANDOM_DIR="/tmp/lm_eval_$(date +%s%N)"
mkdir -p "$RANDOM_DIR"
echo "Saving temporary results to $RANDOM_DIR"

# Convert host paths to container paths
# OUTPUT_DIR and OUTPUT_FILE contain host paths, but we need container paths
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
    # Path relative to current working directory - remove SCR prefix and add /workspace
    RELATIVE_PATH="${CONTAINER_OUTPUT_FILE#$SCR}"
    # Remove leading slash if present
    RELATIVE_PATH="${RELATIVE_PATH#/}"
    CONTAINER_OUTPUT_FILE="/workspace/${RELATIVE_PATH}"
    echo "DEBUG: Converted to: $CONTAINER_OUTPUT_FILE"
elif [[ "$CONTAINER_OUTPUT_FILE" == "$PFS_USER_PREFIX"* ]]; then
    echo "DEBUG: Path matches PFS prefix, converting..."
    # /pfs paths under user directory
    RELATIVE_PATH="${CONTAINER_OUTPUT_FILE#$PFS_USER_PREFIX}"
    RELATIVE_PATH="${RELATIVE_PATH#evals/}"
    CONTAINER_OUTPUT_FILE="/workspace/${RELATIVE_PATH}"
    echo "DEBUG: Converted to: $CONTAINER_OUTPUT_FILE"
elif [[ "$CONTAINER_OUTPUT_FILE" == "$USER_SCRATCH_DIR"* ]]; then
    echo "DEBUG: Path matches USER_SCRATCH_DIR prefix, converting..."
    # Direct /scratch paths under user directory
    RELATIVE_PATH="${CONTAINER_OUTPUT_FILE#$USER_SCRATCH_DIR}"
    # Remove leading slash if present
    RELATIVE_PATH="${RELATIVE_PATH#/}"
    CONTAINER_OUTPUT_FILE="/workspace/${RELATIVE_PATH}"
    echo "DEBUG: Converted to: $CONTAINER_OUTPUT_FILE"
else
    echo "DEBUG: No path conversion matched - keeping original path"
fi

# Prepare final output directory inside container
mkdir -p "$(dirname "$CONTAINER_OUTPUT_FILE")"
echo "Final results will be saved to: $CONTAINER_OUTPUT_FILE"

# Set up chat template flags
{% if env_vars.APPLY_CHAT_TEMPLATE == "True" %}
CHAT_TEMPLATE_FLAG="--apply_chat_template"
{% else %}
CHAT_TEMPLATE_FLAG=""
{% endif %}

{% if env_vars.FEWSHOT_AS_MULTITURN == "True" %}
FEWSHOT_AS_MULTITURN_FLAG="--fewshot_as_multiturn"
{% else %}
FEWSHOT_AS_MULTITURN_FLAG=""
{% endif %}

# Backend-specific model configuration
{% if env_vars.BACKEND == "vllm" %}
# vLLM backend configuration
BASE_VLLM_ARGS="pretrained=${MODEL_LOCAL},dtype=auto,download_dir=/project/hf_cache/models,tensor_parallel_size=${TP}"
DEFAULT_VLLM_ARGS="max_model_len=4096,gpu_memory_utilization=0.90"

# Add custom vLLM arguments if provided
{% if env_vars.VLLM_ARGS %}
CUSTOM_VLLM_ARGS="{{ env_vars.VLLM_ARGS }}"
MODEL_ARGS="${BASE_VLLM_ARGS},${DEFAULT_VLLM_ARGS},${CUSTOM_VLLM_ARGS}"
{% else %}
MODEL_ARGS="${BASE_VLLM_ARGS},${DEFAULT_VLLM_ARGS}"
{% endif %}

MODEL_BACKEND="vllm"
echo "Using vLLM backend with args: $MODEL_ARGS"
{% else %}
# HuggingFace backend configuration
MODEL_ARGS="pretrained=${MODEL_LOCAL},device_map=auto,dtype=bfloat16,trust_remote_code=True,attn_implementation=sdpa"
MODEL_BACKEND="hf-auto"
echo "Using HuggingFace backend with args: $MODEL_ARGS"
{% endif %}

# ------- run the eval (point to local model dir) -------
python -m lm_eval \
  --model "$MODEL_BACKEND" \
  --model_args "$MODEL_ARGS" \
  --tasks "{{ env_vars.TASK_LIST }}" \
  --num_fewshot {{ env_vars.NUM_FEWSHOT }} \
  --batch_size auto \
  --output_path "$RANDOM_DIR" \
  $CHAT_TEMPLATE_FLAG \
  $FEWSHOT_AS_MULTITURN_FLAG \
{% if env_vars.LIMIT %}  --limit {{ env_vars.LIMIT }} \
{% endif %}  --log_samples

echo "Moving temporary results from $RANDOM_DIR to $CONTAINER_OUTPUT_FILE"
find "$RANDOM_DIR" -name "results_*.json" -exec mv {} "$CONTAINER_OUTPUT_FILE" \;
rm -rf "$RANDOM_DIR"

# Clean up the temporary lm-eval-harness directory
echo "Cleaning up temporary lm-eval directory: $EVAL_HARNESS_DIR"
rm -rf "$EVAL_HARNESS_DIR"

echo "== results saved to $CONTAINER_OUTPUT_FILE =="
ls -l "$CONTAINER_OUTPUT_FILE" || true
'
