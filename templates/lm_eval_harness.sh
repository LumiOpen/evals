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
export SCR="$(pwd -P)"                     # SCR = scratch directory, will be /workspace in container (resolve symlinks)
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
    --env HF_HOME=/project/hf_cache \
    --env HUGGINGFACE_HUB_CACHE=/project/hf_cache/hub \
    --env TRANSFORMERS_CACHE=/project/hf_cache/models \
    --env HF_DATASETS_CACHE=/project/hf_cache/datasets \
    --env XDG_CACHE_HOME=/project/hf_cache/xdg \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    "$IMG" bash -c '
set -euo pipefail
umask 002

# Force HOME=/tmp so aiter builds ephemerally and disappears with the job
export HOME=/tmp

PYTHON_BIN="/opt/miniconda3/envs/pytorch/bin/python"
export PYTHON_BIN

# ---- env & caches (disable xet + telemetry) ----
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1

export PATH="/opt/rocm/llvm/bin:/opt/rocm/bin:/opt/miniconda3/envs/pytorch/bin:/usr/local/bin:/usr/bin:/bin"
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext
export TORCHINDUCTOR_CACHE_DIR=/project/hf_cache/torchinductor
export VLLM_COMPILER_CACHE_DIR=/project/hf_cache/vllm-compile
export TRITON_CACHE_DIR=/project/hf_cache/triton

export VLLM_USE_V1=1
export VLLM_TARGET_DEVICE=rocm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HIP_ARCHITECTURES=gfx90a

# Avoid the 1-GPU trap - ensure PyTorch uses all allocated GPUs
unset HIP_VISIBLE_DEVICES

mkdir -p /project/hf_cache/{hub,models,datasets,torchinductor,xdg,vllm-compile,triton} \
         /dev/shm/torch_ext "$HOME/.aiter/jit/build" "$HOME/.aiter/jit/install" \
         /tmp/tools

export CC=/opt/rocm/llvm/bin/clang
export CXX=/opt/rocm/llvm/bin/clang++

# Make sure ninja is available
${PYTHON_BIN} -m pip -q install --user -U ninja || true

# ------- write helper: stage_aiter.py (NO stdin execution) -------
cat > /tmp/tools/stage_aiter.py <<PY
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

# Stage aiter
${PYTHON_BIN} /tmp/tools/stage_aiter.py

export PYTHONPATH="$HOME/.aiter/jit/install:${PYTHONPATH-}"

# Prepare patched config override when rope_scaling is boolean
CONFIG_OVERRIDE_PATH="/tmp/config-override-${SLURM_JOB_ID:-$$}.json"
cat > /tmp/tools/prepare_model_config.py <<'PY'
import json, os, urllib.request

model_id = os.environ.get("MODEL_ID", "")
target = os.environ.get("CONFIG_OVERRIDE_PATH")
if not model_id or model_id.count("/") != 1 or not target:
    raise SystemExit(0)

org, name = model_id.split("/", 1)
url = f"https://huggingface.co/{org}/{name}/resolve/main/config.json"
headers = {}
token = os.environ.get("HF_TOKEN")
if token:
    headers["Authorization"] = f"Bearer {token}"

req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=30) as resp, open(target, "wb") as out:
        out.write(resp.read())
except Exception as exc:
    print(f"[config] skip override: {exc}")
    if os.path.exists(target):
        os.remove(target)
    raise SystemExit(0)

with open(target, "r", encoding="utf-8") as fh:
    config = json.load(fh)

rope_scaling = config.get("rope_scaling")
if isinstance(rope_scaling, bool):
    config.pop("rope_scaling", None)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(config, fh)
else:
    os.remove(target)
PY

MODEL_CONFIG_OVERRIDE=""
if ${PYTHON_BIN} /tmp/tools/prepare_model_config.py && [ -f "$CONFIG_OVERRIDE_PATH" ]; then
    MODEL_CONFIG_OVERRIDE="$CONFIG_OVERRIDE_PATH"
fi

# Install requested transformers package into a writable site-packages tree and prepend to PYTHONPATH
TRANSFORMERS_SITE="/tmp/python-packages-${SLURM_JOB_ID:-$$}"
rm -rf "$TRANSFORMERS_SITE"
mkdir -p "$TRANSFORMERS_SITE"
${PYTHON_BIN} -m pip install -q -U --no-deps --target "$TRANSFORMERS_SITE" "{{ env_vars.TRANSFORMERS_PACKAGE }}"
${PYTHON_BIN} -m pip install -q -U --no-deps --target "$TRANSFORMERS_SITE" "{{ env_vars.TOKENIZERS_PACKAGE }}"
export PYTHONPATH="$TRANSFORMERS_SITE:${PYTHONPATH-}"

# ------- get LUMI harness (puts it first on sys.path) -------
EVAL_HARNESS_DIR="/tmp/lm-eval"

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
export PYTHONPATH="$EVAL_HARNESS_DIR:${PYTHONPATH-}"

# Create a temporary directory for lm_eval output
RANDOM_DIR="/tmp/lm_eval_$(date +%s%N)"
mkdir -p "$RANDOM_DIR"

OUTPUT_FILE="{{ env_vars.OUTPUT_FILE }}"
if [[ "$OUTPUT_FILE" == "$SCR"* ]]; then
  # map host path under $SCR to the /workspace bind mount
  OUTPUT_FILE="/workspace${OUTPUT_FILE#$SCR}"
elif [[ "$OUTPUT_FILE" != /* ]]; then
  OUTPUT_FILE="/workspace/$OUTPUT_FILE"
fi
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Set up chat template flags
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
MODEL_ARGS="pretrained=${MODEL_ID},dtype=auto,download_dir=/project/hf_cache/models,tensor_parallel_size=${TP},max_model_len=4096,gpu_memory_utilization=0.90"
if [[ -n "$MODEL_CONFIG_OVERRIDE" ]]; then
  MODEL_ARGS="${MODEL_ARGS},hf_config_path=${MODEL_CONFIG_OVERRIDE}"
fi
{% if env_vars.MODEL_ARGS %}
MODEL_ARGS="${MODEL_ARGS},{{ env_vars.MODEL_ARGS }}"
{% endif %}
{% elif env_vars.BACKEND == "dummy" %}
MODEL_BACKEND="dummy"
MODEL_ARGS="pretrained={{ env_vars.MODEL }}"
{% else %}
MODEL_BACKEND="hf-auto"
MODEL_ARGS="pretrained=${MODEL_ID},device_map=auto,dtype=bfloat16,trust_remote_code=True,attn_implementation=sdpa"
{% if env_vars.MODEL_ARGS %}
MODEL_ARGS="${MODEL_ARGS},{{ env_vars.MODEL_ARGS }}"
{% endif %}
{% endif %}

# Run eval
BATCH_SIZE="{% if env_vars.BATCH_SIZE %}{{ env_vars.BATCH_SIZE }}{% elif env_vars.BACKEND == "vllm" %}auto{% else %}4{% endif %}"

${PYTHON_BIN} -m lm_eval \
  --model "$MODEL_BACKEND" \
  --model_args "$MODEL_ARGS" \
  --tasks "{{ env_vars.TASK_LIST }}" \
  --num_fewshot {{ env_vars.NUM_FEWSHOT }} \
  --batch_size "$BATCH_SIZE" \
  --output_path "$RANDOM_DIR" \
  $CHAT_TEMPLATE_FLAG \
  $FEWSHOT_AS_MULTITURN_FLAG \
{% if env_vars.LIMIT %}  --limit {{ env_vars.LIMIT }} \
{% endif %}  --log_samples \
{% if env_vars.LM_EVAL_ARGS %}  {{ env_vars.LM_EVAL_ARGS }}
{% endif %}

find "$RANDOM_DIR" -name "results_*.json" -exec mv {} "$OUTPUT_FILE" \;
rm -rf "$RANDOM_DIR"
'
