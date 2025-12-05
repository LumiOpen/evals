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
${PYTHON_BIN} -m pip install --target "$SITE_PACKAGES" -q -U ninja || true

# Skip aiter setup for this container (not needed for lumi-multitorch)
echo "[aiter] Skipping aiter setup for this container"

# Install EuroEval without torch/vllm deps - use the container ROCm versions
echo "Installing EuroEval to $SITE_PACKAGES (using the container ROCm torch/vllm)..."
${PYTHON_BIN} -m pip install --target "$SITE_PACKAGES" --no-deps euroeval

# Install other EuroEval dependencies (excluding torch, vllm, transformers, accelerate which are in container)
${PYTHON_BIN} -m pip install --target "$SITE_PACKAGES" \
    bert-score click cloudpickle datasets demjson3 evaluate levenshtein litellm \
    more-itertools numpy ollama pandas peft protobuf pydantic pyinfer python-dotenv \
    rouge-score sacremoses scikit-learn sentencepiece seqeval setuptools tenacity \
    termcolor huggingface-hub
echo "EuroEval install complete"

# Verify we are using the container ROCm torch
echo "Checking torch version..."
${PYTHON_BIN} -c "import torch; print(\"torch:\", torch.__version__, \"| HIP:\", torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\")"

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

# Write wrapper script that patches imports before loading euroeval
cat > /tmp/run_euroeval.py <<WRAPPER
import importlib.util
import sys

# Patch find_spec to hide flash_attn from EuroEval check
_orig = importlib.util.find_spec
def _patched(name, *a, **kw):
    return None if name == "flash_attn" else _orig(name, *a, **kw)
importlib.util.find_spec = _patched

# Patch vllm.sampling_params if needed (for older vLLM versions)
try:
    import vllm.sampling_params as sp
    if not hasattr(sp, "StructuredOutputsParams"):
        class StructuredOutputsParams:
            def __init__(self, *args, **kwargs):
                pass
        sp.StructuredOutputsParams = StructuredOutputsParams
        print("[patch] Added stub StructuredOutputsParams to vllm.sampling_params")
except ImportError:
    print("[patch] vllm not available, skipping patch")

# Set argv and run
sys.argv = ["euroeval"] + sys.argv[1:]
from euroeval.cli import benchmark
benchmark()
WRAPPER

${PYTHON_BIN} /tmp/run_euroeval.py $EUROEVAL_ARGS

# Copy results to output location
if [ -f "euroeval_benchmark_results.jsonl" ]; then
    cp "euroeval_benchmark_results.jsonl" "$OUTPUT_FILE"
    echo "Results saved to: $OUTPUT_FILE"
fi
'
