#!/bin/bash
#SBATCH --job-name=longppl
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclusive

# LongPPL cross-lingual key-token perplexity evaluation on TensorWave.
# Uses LumiOpen's lm-eval fork with crosslingual-longppl tasks (en + fi).
#
# Usage:
#   sbatch --export=ALL,MODEL=/shared_silo/scratch/models/Meta-Llama-3.1-8B eval_longppl.sbatch
#   sbatch --export=ALL,MODEL=LumiOpen/Poro-2-8B-base,ROPE_SCALING=true eval_longppl.sbatch
#   sbatch --export=ALL,MODEL=LumiOpen/Llama-Poro-2-Long-Base eval_longppl.sbatch
#
# Optional overrides:
#   SCRATCH       — scratch root (default: /shared_silo/scratch/$USER)
#   ROPE_SCALING  — set to "true" for models needing RoPE extension (default: false)
#   TASKS         — comma-separated lm-eval tasks (default: en + fi)
#   IMG           — container image path

set -euo pipefail

MODEL="${MODEL:?Set MODEL via --export=ALL,MODEL=...}"
SCRATCH="${SCRATCH:-/shared_silo/scratch/$USER}"
ROPE_SCALING="${ROPE_SCALING:-false}"
LMEVAL_DIR="${SCRATCH}/lm-eval-longppl"
IMG="${IMG:-/shared_silo/scratch/containers/rocm_vllm_rocm7.0.0_vllm_0.11.2_20251210.sif}"
OUTDIR="${SCRATCH}/eval_results/longppl"

MODEL_ARGS="pretrained=${MODEL},dtype=bfloat16,max_length=32768,allow_truncation=true"
if [ "$ROPE_SCALING" = "true" ]; then
    MODEL_ARGS="${MODEL_ARGS},auto_rope_scaling=true"
fi

TASKS="${TASKS:-crosslingual_longppl_en,crosslingual_longppl_fi}"

echo "=== LongPPL evaluation ==="
echo "Job ID:      $SLURM_JOBID"
echo "Model:       $MODEL"
echo "Model args:  $MODEL_ARGS"
echo "RoPE scale:  $ROPE_SCALING"
echo "Tasks:       $TASKS"
echo "Output:      $OUTDIR"
echo "Start:       $(date)"

mkdir -p "$OUTDIR"

apptainer exec --rocm \
    -B /shared_silo/scratch:/shared_silo/scratch:rw \
    -B /usr/share/libdrm:/usr/share/libdrm:ro \
    "$IMG" bash -c "
set -euo pipefail

export HIP_VISIBLE_DEVICES=0
unset CUDA_VISIBLE_DEVICES
export HF_HOME=${SCRATCH}/hf_home
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton-\${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-\${SLURM_JOB_ID}
export PYTHONUSERBASE=/tmp/pip-longppl-\${SLURM_JOB_ID}

pip install --user -e $LMEVAL_DIR --quiet 2>&1 | tail -5
pip install --user 'transformers>=4.57.1' --quiet 2>&1 | tail -3

echo '--- Running LongPPL ---'
python -m lm_eval \\
    --model longppl_hf \\
    --model_args $MODEL_ARGS \\
    --tasks $TASKS \\
    --batch_size 1 \\
    --output_path $OUTDIR \\
    --log_samples
"

echo "End: $(date)"
echo "Results in: $OUTDIR"
