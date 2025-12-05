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
export PRJ="/scratch/{{ slurm_config.account }}"
export SCR="$(pwd -P)"
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

mkdir -p /project/hf_cache/{hub,models,datasets,torchinductor,xdg,vllm-compile,triton,euroeval} \
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

# Install EuroEval
${PYTHON_BIN} -m pip install --user -q euroeval

# Install specific transformers version
${PYTHON_BIN} -m pip install --user -q -U transformers=={{ env_vars.TRANSFORMERS_VERSION }}

# Remap MODEL_ID from host paths to container paths if needed
if [[ "$MODEL_ID" == /scratch/{{ slurm_config.account }}/* ]]; then
  MODEL_ID="/project${MODEL_ID#/scratch/{{ slurm_config.account }}}"
elif [[ "$MODEL_ID" == "$SCR"/* ]]; then
  MODEL_ID="/workspace${MODEL_ID#$SCR}"
fi

OUTPUT_FILE="{{ env_vars.OUTPUT_FILE }}"
if [[ "$OUTPUT_FILE" == "$SCR"* ]]; then
  OUTPUT_FILE="/workspace${OUTPUT_FILE#$SCR}"
elif [[ "$OUTPUT_FILE" != /* ]]; then
  OUTPUT_FILE="/workspace/$OUTPUT_FILE"
fi
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Build euroeval command arguments
EUROEVAL_ARGS="--model ${MODEL_ID}"
EUROEVAL_ARGS="$EUROEVAL_ARGS --cache-dir /project/hf_cache/euroeval"

{% if env_vars.LANGUAGE %}
EUROEVAL_ARGS="$EUROEVAL_ARGS --language {{ env_vars.LANGUAGE }}"
{% endif %}

{% if env_vars.TASK %}
EUROEVAL_ARGS="$EUROEVAL_ARGS --task {{ env_vars.TASK }}"
{% endif %}

{% if env_vars.DATASET %}
EUROEVAL_ARGS="$EUROEVAL_ARGS --dataset {{ env_vars.DATASET }}"
{% endif %}

{% if env_vars.FEW_SHOT == "False" %}
EUROEVAL_ARGS="$EUROEVAL_ARGS --zero-shot"
{% endif %}

{% if env_vars.EVALUATE_TEST_SPLIT == "True" %}
EUROEVAL_ARGS="$EUROEVAL_ARGS --evaluate-test-split"
{% endif %}

{% if env_vars.TRUST_REMOTE_CODE == "True" %}
EUROEVAL_ARGS="$EUROEVAL_ARGS --trust-remote-code"
{% endif %}

{% if env_vars.NUM_ITERATIONS %}
EUROEVAL_ARGS="$EUROEVAL_ARGS --num-iterations {{ env_vars.NUM_ITERATIONS }}"
{% endif %}

{% if env_vars.BATCH_SIZE %}
EUROEVAL_ARGS="$EUROEVAL_ARGS --finetuning-batch-size {{ env_vars.BATCH_SIZE }}"
{% endif %}

# GPU memory utilization
EUROEVAL_ARGS="$EUROEVAL_ARGS --gpu-memory-utilization {{ env_vars.GPU_MEMORY_UTILIZATION | default(0.85) }}"

# Always use verbose and save results
EUROEVAL_ARGS="$EUROEVAL_ARGS --verbose --save-results"

echo "Running EuroEval with arguments: $EUROEVAL_ARGS"

# Run EuroEval with find_spec patched to hide flash_attn
# (EuroEval uses find_spec to check if flash_attn exists, not import)
${PYTHON_BIN} -c "
import importlib.util
_orig = importlib.util.find_spec
def _patched(name, *a, **kw):
    return None if name == 'flash_attn' else _orig(name, *a, **kw)
importlib.util.find_spec = _patched

import sys
sys.argv = ['euroeval'] + '''$EUROEVAL_ARGS'''.split()
from euroeval.cli import main
main()
"

# Copy results to output location
if [ -f "euroeval_benchmark_results.jsonl" ]; then
    cp "euroeval_benchmark_results.jsonl" "$OUTPUT_FILE"
    echo "Results saved to: $OUTPUT_FILE"
fi
'
