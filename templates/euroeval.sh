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

export IMG="/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t28-20251128_145346/lumi-multitorch-full-u24r64f21m43t28-20251128_145346.sif"
export PRJ="$(readlink -f /scratch/project_462000353)"   # always use 353 for cache, resolve symlinks for singularity bind
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

srun -A "$ACC" -p "{{ slurm_config.partition }}" -N "$N_NODES" -n1 -t "{{ slurm_config.time }}" --gpus-per-task="$GPUS" \
  singularity exec --rocm \
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
    --env VLLM_USE_V1=1 \
    --env VLLM_TARGET_DEVICE=rocm \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env HIP_ARCHITECTURES=gfx90a \
    "$IMG" bash -c '
set -euo pipefail
umask 002

# Force HOME=/tmp so aiter builds ephemerally and disappears with the job
export HOME=/tmp

# Set up writable site-packages for pip installs
# IMPORTANT: Container packages (ROCm torch) must come FIRST in PYTHONPATH
# so they take precedence over pip-installed CUDA packages
export SITE_PACKAGES=/project/hf_cache/python_user/lib/python3.12/site-packages
mkdir -p "$SITE_PACKAGES"
export PYTHONPATH="/opt/venv/lib/python3.12/site-packages:$SITE_PACKAGES:${PYTHONPATH:-}"

# Find Python - different containers have different paths
if [ -x /opt/miniconda3/envs/pytorch/bin/python ]; then
    PYTHON_BIN="/opt/miniconda3/envs/pytorch/bin/python"
elif [ -x /opt/venv/bin/python ]; then
    PYTHON_BIN="/opt/venv/bin/python"
else
    echo "ERROR: No suitable Python found in container"
    exit 1
fi
export PYTHON_BIN
echo "Using Python: $PYTHON_BIN"
echo "Site packages: $SITE_PACKAGES"

# ---- env & caches (disable xet + telemetry) ----
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1

export PATH="/opt/rocm/llvm/bin:/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin"
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext
export TORCHINDUCTOR_CACHE_DIR=/project/hf_cache/torchinductor
export VLLM_COMPILER_CACHE_DIR=/project/hf_cache/vllm-compile
export TRITON_CACHE_DIR=/project/hf_cache/triton

# vLLM V1 works on ROCm when VLLM_ATTENTION_BACKEND is NOT set (auto-detect)
# DO NOT set VLLM_ATTENTION_BACKEND - EuroEval sets FLASHINFER which breaks ROCm
export VLLM_USE_V1=1
export VLLM_TARGET_DEVICE=rocm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HIP_ARCHITECTURES=gfx90a

# Avoid the 1-GPU trap - ensure PyTorch uses all allocated GPUs
# Also unset ROCR_VISIBLE_DEVICES as vLLM errors if it's set
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES

mkdir -p /project/hf_cache/{hub,models,datasets,torchinductor,xdg,vllm-compile,triton,euroeval} \
         /dev/shm/torch_ext "$HOME/.aiter/jit/build" "$HOME/.aiter/jit/install" \
         /tmp/tools

export CC=/opt/rocm/llvm/bin/clang
export CXX=/opt/rocm/llvm/bin/clang++

# Make sure ninja is available
${PYTHON_BIN} -m pip install --target "$SITE_PACKAGES" -q -U ninja || true

# Skip aiter setup for this container (not needed for lumi-multitorch)
echo "[aiter] Skipping aiter setup for this container"

# Install EuroEval without torch/vllm deps - use the container ROCm versions
echo "Installing EuroEval to $SITE_PACKAGES (using the container ROCm torch/vllm)..."
${PYTHON_BIN} -m pip install --target "$SITE_PACKAGES" --no-deps euroeval

# Install other EuroEval dependencies with --no-deps to avoid pulling in CUDA torch
# These are the direct deps that do NOT require torch/vllm/transformers (those are in container)
${PYTHON_BIN} -m pip install --target "$SITE_PACKAGES" --no-deps \
    bert-score click cloudpickle datasets demjson3 evaluate levenshtein litellm \
    more-itertools numpy ollama pandas peft protobuf pydantic pyinfer python-dotenv \
    rouge-score sacremoses scikit-learn sentencepiece seqeval setuptools tenacity \
    termcolor huggingface-hub

# Install sub-dependencies that are safe (no torch)
${PYTHON_BIN} -m pip install --target "$SITE_PACKAGES" \
    filelock fsspec packaging pyyaml requests tqdm regex safetensors tokenizers \
    aiohttp httpx jinja2 typing-extensions annotated-types pydantic-core \
    pyarrow dill multiprocess xxhash rapidfuzz joblib threadpoolctl scipy \
    absl-py nltk six tabulate grpcio openai tiktoken jsonschema importlib-metadata
echo "EuroEval install complete"

# Verify we are using the container ROCm torch
echo "Checking torch version..."
${PYTHON_BIN} -c "import torch; print(\"torch:\", torch.__version__, \"| HIP:\", torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\")"

# Remap MODEL_ID from host paths to container paths if needed
if [[ "$MODEL_ID" == /scratch/project_462000353/* ]]; then
  MODEL_ID="/project${MODEL_ID#/scratch/project_462000353}"
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

# Write wrapper script that patches imports before loading euroeval
cat > /tmp/run_euroeval.py <<WRAPPER
import os
import importlib.util
import sys

# Patch find_spec to hide flash_attn from EuroEval
_orig_find_spec = importlib.util.find_spec
def _patched_find_spec(name, *a, **kw):
    if name == "flash_attn":
        return None
    return _orig_find_spec(name, *a, **kw)
importlib.util.find_spec = _patched_find_spec

# Patch shutil.which to return a fake nvcc path for ROCm
# EuroEval checks for nvcc but we are using ROCm/HIP which does not need nvcc
import shutil
_orig_which = shutil.which
def _patched_which(cmd, *a, **kw):
    if cmd == "nvcc":
        return "/opt/rocm/bin/hipcc"  # Return hipcc as nvcc equivalent
    return _orig_which(cmd, *a, **kw)
shutil.which = _patched_which

print("[patch] flash_attn hidden, nvcc check bypassed for ROCm")

# Import euroeval - this will set VLLM_USE_V1=1 and VLLM_ATTENTION_BACKEND=FLASHINFER
# We need to import it first, then override those settings before vllm is loaded
import euroeval  # This triggers euroeval.__init__ which sets the env vars

# CRITICAL: Override EuroEval settings for ROCm compatibility
# EuroEval sets VLLM_ATTENTION_BACKEND=FLASHINFER which is CUDA-only
# On ROCm, we must NOT set this - let vLLM auto-detect the right backend
# VLLM_USE_V1=1 works fine on ROCm when attention backend is not forced
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_TARGET_DEVICE"] = "rocm"
if "VLLM_ATTENTION_BACKEND" in os.environ:
    del os.environ["VLLM_ATTENTION_BACKEND"]  # Remove FLASHINFER, let vLLM auto-detect
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
print("[patch] ROCm vLLM env: VLLM_USE_V1=" + os.environ.get("VLLM_USE_V1", "") + ", VLLM_ATTENTION_BACKEND=" + os.environ.get("VLLM_ATTENTION_BACKEND", "auto-detect"))

# Set argv and run
sys.argv = ["euroeval"] + sys.argv[1:]
from euroeval.cli import benchmark
benchmark()
WRAPPER

# Export vLLM env vars in shell BEFORE Python starts
# Do NOT set VLLM_ATTENTION_BACKEND - let vLLM auto-detect for ROCm
export VLLM_USE_V1=1
export VLLM_TARGET_DEVICE=rocm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "[shell] VLLM_USE_V1=$VLLM_USE_V1 VLLM_TARGET_DEVICE=$VLLM_TARGET_DEVICE"

${PYTHON_BIN} /tmp/run_euroeval.py $EUROEVAL_ARGS

# Copy results to output location
if [ -f "euroeval_benchmark_results.jsonl" ]; then
    cp "euroeval_benchmark_results.jsonl" "$OUTPUT_FILE"
    echo "Results saved to: $OUTPUT_FILE"
fi
'
