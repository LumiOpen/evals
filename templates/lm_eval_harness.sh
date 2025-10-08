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
    --env USER="$USER" \
    --env DEVICE_MAP=balanced_low_0 \
    --env PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    --env HF_TOKEN="$HF_TOKEN" \
    --env HF_HOME=/project/hf_cache \
    --env HUGGINGFACE_HUB_CACHE=/project/hf_cache/hub \
    --env TRANSFORMERS_CACHE=/project/hf_cache/models \
    --env HF_DATASETS_CACHE=/project/hf_cache/datasets \
    --env XDG_CACHE_HOME=/project/hf_cache/xdg \
    --env PYTHONUSERBASE=/project/python_user \
    --env TMPDIR=/tmp \
    --env PIP_CACHE_DIR=/project/hf_cache/pip \
    --env CULTURAL_ROBUSTNESS_CACHE_DIR=/workspace/cultural_robustness_cache \
    --env MODEL_NAME="${MODEL_ID//\//-}" \
    --env EMBEDDING_DEVICE="{{ env_vars.EMBEDDING_DEVICE }}" \
    {% if env_vars.EMBEDDING_MODEL %}--env EMBEDDING_MODEL="{{ env_vars.EMBEDDING_MODEL }}" \
    {% endif %}{% if env_vars.EVAL_LANGUAGES %}--env EVAL_LANGUAGES="{{ env_vars.EVAL_LANGUAGES }}" \
    {% endif %}"$IMG" bash -c '
set -euo pipefail
umask 002

# ---- model/topology (now variables) ----
MODEL_ID="${MODEL_ID:?MODEL_ID not set}"
TP="${TP:-32}"
MODEL_SAFE="${MODEL_ID//\//-}"
PREFETCH_LOCAL_DIR="/project/hf_cache/models/${MODEL_SAFE}"
MODEL_LOCAL="${PREFETCH_LOCAL_DIR}"

# ---- env & caches (disable xet + telemetry) ----
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1

export PATH="$PYTHONUSERBASE/bin:$HOME/.local/bin:/opt/miniconda3/envs/pytorch/bin:/opt/rocm/llvm/bin:/opt/rocm/bin:/usr/bin:/bin"
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

mkdir -p /project/hf_cache/{hub,models,datasets,torchinductor,xdg,vllm-compile,pip} \
         /project/python_user \
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
python -m pip install --user --no-cache-dir transformers==4.56.0
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
{% endif %}

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
{% if env_vars.BACKEND != "dummy" %}
python /workspace/tools/prefetch.py
{% endif %}

{% if env_vars.BACKEND != "dummy" %}
# Model is prefetched, but allow datasets to be downloaded as needed
# Note: Datasets will be cached automatically for subsequent runs
{% endif %}

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
pip install --user --no-cache-dir transformers==4.56.0
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
CONTAINER_LOG_SAMPLES="{{ env_vars.LOG_SAMPLES }}"
LOG_SAMPLES_FLAG=""
if [[ -n "$CONTAINER_LOG_SAMPLES" ]]; then
    echo "DEBUG: Original LOG_SAMPLES: $CONTAINER_LOG_SAMPLES"
    if [[ "$CONTAINER_LOG_SAMPLES" == "$SCR"* ]]; then
        RELATIVE_PATH="${CONTAINER_LOG_SAMPLES#$SCR}"
        RELATIVE_PATH="${RELATIVE_PATH#/}"
        CONTAINER_LOG_SAMPLES="/workspace/${RELATIVE_PATH}"
    elif [[ "$CONTAINER_LOG_SAMPLES" == "$PFS_USER_PREFIX"* ]]; then
        RELATIVE_PATH="${CONTAINER_LOG_SAMPLES#$PFS_USER_PREFIX}"
        RELATIVE_PATH="${RELATIVE_PATH#evals/}"
        CONTAINER_LOG_SAMPLES="/workspace/${RELATIVE_PATH}"
    elif [[ "$CONTAINER_LOG_SAMPLES" == "$USER_SCRATCH_DIR"* ]]; then
        RELATIVE_PATH="${CONTAINER_LOG_SAMPLES#$USER_SCRATCH_DIR}"
        RELATIVE_PATH="${RELATIVE_PATH#/}"
        RELATIVE_PATH="${RELATIVE_PATH#evals/}"
        CONTAINER_LOG_SAMPLES="/workspace/${RELATIVE_PATH}"
    fi
    echo "DEBUG: LOG_SAMPLES container path: $CONTAINER_LOG_SAMPLES"
    mkdir -p "$(dirname "$CONTAINER_LOG_SAMPLES")"
    LOG_SAMPLES_FLAG="--log_samples"
fi

mkdir -p "$(dirname "$CONTAINER_OUTPUT_FILE")"
echo "Final results will be saved to: $CONTAINER_OUTPUT_FILE"

# Set OUTPUT_DIR for cluster outputs (same directory as results file)
export OUTPUT_DIR="$(dirname "$CONTAINER_OUTPUT_FILE")"
echo "OUTPUT_DIR set to: $OUTPUT_DIR"

# Set up chat template flags
{% if env_vars.APPLY_CHAT_TEMPLATE == "True" %}
CHAT_TEMPLATE_FLAG="--apply_chat_template"
echo "✓ Chat template will be applied (APPLY_CHAT_TEMPLATE={{ env_vars.APPLY_CHAT_TEMPLATE }})"
{% else %}
CHAT_TEMPLATE_FLAG=""
echo "✗ Chat template will NOT be applied (APPLY_CHAT_TEMPLATE={{ env_vars.APPLY_CHAT_TEMPLATE }})"
{% endif %}

{% if env_vars.FEWSHOT_AS_MULTITURN == "True" %}
FEWSHOT_AS_MULTITURN_FLAG="--fewshot_as_multiturn"
echo "✓ Fewshot as multiturn enabled"
{% else %}
FEWSHOT_AS_MULTITURN_FLAG=""
echo "✗ Fewshot as multiturn disabled"
{% endif %}

# Backend-specific model configuration
{% if env_vars.BACKEND == "vllm" %}
# vLLM backend configuration
BASE_VLLM_ARGS="pretrained=${MODEL_LOCAL},dtype={{ env_vars.DTYPE }},download_dir=/project/hf_cache/models,tensor_parallel_size=${TP}"
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
{% elif env_vars.BACKEND == "dummy" %}
MODEL_BACKEND="dummy"
MODEL_ARGS="pretrained={{ env_vars.MODEL }}"
echo "Using lm_eval dummy backend (no model weights, {{ env_vars.MODEL }} tokenizer for chat templates)"
{% else %}
# HuggingFace backend configuration
{% if env_vars.MAX_MEMORY_JSON %}
# Use JSON format with max_memory dict (integer keys for accelerate)
cat > /tmp/build_model_args_$$.py <<PYEOF
import json
import ast
# Convert string keys to integers for accelerate compatibility
max_memory_str = {{ env_vars.MAX_MEMORY_JSON }}
max_memory = {int(k): v for k, v in max_memory_str.items()}
# Build model_args with integer keys in max_memory
model_args = {
    "pretrained": "${MODEL_LOCAL}",
    "device_map": "{{ env_vars.DEVICE_MAP }}",
    "dtype": "{{ env_vars.DTYPE }}",
    "trust_remote_code": True,
    "attn_implementation": "eager",
    "add_bos_token": True,
    "low_cpu_mem_usage": True,
    "max_memory": max_memory
}
# Use repr() which preserves integer keys, then convert to JSON-like format
# that lm-eval can parse while keeping integer keys
print(json.dumps(model_args, default=str))
PYEOF
# Parse with Python to handle integer keys properly
MODEL_ARGS='pretrained=${MODEL_LOCAL},device_map={{ env_vars.DEVICE_MAP }},dtype={{ env_vars.DTYPE }},trust_remote_code=True,attn_implementation=eager,add_bos_token=True,low_cpu_mem_usage=True,max_memory='$(python3 -c "import json; m={{ env_vars.MAX_MEMORY_JSON }}; print('{' + ','.join(f'{k}:\"{v}\"' for k,v in m.items()) + '}')")
rm -f /tmp/build_model_args_$$.py
echo "Using HuggingFace backend with max_memory per GPU"
{% else %}
MODEL_ARGS="pretrained=${MODEL_LOCAL},device_map={{ env_vars.DEVICE_MAP }},dtype={{ env_vars.DTYPE }},trust_remote_code=True,attn_implementation=eager,add_bos_token=True,low_cpu_mem_usage=True"
echo "Using HuggingFace backend with device_map={{ env_vars.DEVICE_MAP }}"
{% endif %}
MODEL_BACKEND="hf-auto"
echo "MODEL_ARGS: $MODEL_ARGS"
{% endif %}

# ------- run the eval (point to local model dir) -------
# Use accelerate for multi-GPU data parallelism
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [ "$NUM_GPUS" -gt 1 ]; then
  echo "Using accelerate with $NUM_GPUS processes for data parallelism"
  # Remove device_map from MODEL_ARGS when using accelerate (it handles device placement)
  MODEL_ARGS=$(echo "$MODEL_ARGS" | sed 's/,device_map=[^,]*//g')
  LAUNCHER="accelerate launch --num_processes $NUM_GPUS --multi_gpu -m"
else
  echo "Using single GPU"
  LAUNCHER="python -m"
fi

$LAUNCHER lm_eval \
  --model "$MODEL_BACKEND" \
  --model_args "$MODEL_ARGS" \
  --tasks "{{ env_vars.TASK_LIST }}" \
  --num_fewshot {{ env_vars.NUM_FEWSHOT }} \
  --batch_size ${TP} \
  --output_path "$RANDOM_DIR" \
{% if env_vars.USE_CACHE %}  --use_cache "{{ env_vars.USE_CACHE }}" \
{% endif %}  $CHAT_TEMPLATE_FLAG \
  $FEWSHOT_AS_MULTITURN_FLAG \
{% if env_vars.LIMIT %}  --limit {{ env_vars.LIMIT }} \
{% endif %}  $LOG_SAMPLES_FLAG

if [[ -n "$CONTAINER_LOG_SAMPLES" ]]; then
  if [[ -d "$RANDOM_DIR/per_sample" ]]; then
    rm -rf "$CONTAINER_LOG_SAMPLES"
    mv "$RANDOM_DIR/per_sample" "$CONTAINER_LOG_SAMPLES"
    echo "Per-sample outputs moved to $CONTAINER_LOG_SAMPLES"
  else
    echo "WARNING: Requested per-sample logging but no per_sample directory was generated"
  fi
fi

echo "Moving temporary results from $RANDOM_DIR to $CONTAINER_OUTPUT_FILE"
find "$RANDOM_DIR" -name "results_*.json" -exec mv {} "$CONTAINER_OUTPUT_FILE" \;
rm -rf "$RANDOM_DIR"

# Clean up the temporary lm-eval-harness directory
echo "Cleaning up temporary lm-eval directory: $EVAL_HARNESS_DIR"
rm -rf "$EVAL_HARNESS_DIR"

echo "== results saved to $CONTAINER_OUTPUT_FILE =="
ls -l "$CONTAINER_OUTPUT_FILE" || true
'
