#!/bin/bash
#SBATCH --job-name=helmet
#SBATCH --partition=amd-tw-verification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --exclusive

# HELMET 128K long-context evaluation on TensorWave (MI325X).
# Runs all 7 standard HELMET categories with vLLM (TP=8).
#
# Usage:
#   sbatch --export=ALL,MODEL=/shared_silo/scratch/models/Meta-Llama-3.1-8B eval_helmet.sbatch
#   sbatch --export=ALL,MODEL=LumiOpen/Llama-Poro-2-Long-Base eval_helmet.sbatch
#
# Optional overrides:
#   SHARED    — shared data root (default: /shared_silo/scratch/shared)
#   SCRATCH   — per-user scratch (default: /shared_silo/scratch/$USER)
#   CONFIGS   — space-separated list of HELMET configs (default: all 7)
#   TAG       — output tag (default: v1)
#   SEED      — random seed (default: 42)
#   IMG       — container image path

set -euo pipefail

MODEL="${MODEL:?Set MODEL via --export=ALL,MODEL=...}"
SHARED="${SHARED:-/shared_silo/scratch/shared}"
SCRATCH="${SCRATCH:-/shared_silo/scratch/$USER}"
HELMET_DIR="${SHARED}/HELMET"
IMG="${IMG:-/shared_silo/scratch/containers/rocm_vllm_rocm7.0.0_vllm_0.11.2_20251210.sif}"

MODEL_SHORT=$(basename "$MODEL")
TAG="${TAG:-v1}"
SEED="${SEED:-42}"
OUTDIR="${SCRATCH}/helmet_output/${MODEL_SHORT}"

CONFIGS=(${CONFIGS:-recall.yaml rag.yaml longqa.yaml summ.yaml icl.yaml rerank.yaml cite.yaml})

# Detect base vs instruct/chat model
OPTIONS=""
shopt -s nocasematch
if ! [[ $MODEL_SHORT =~ (chat|instruct|it$) ]]; then
    OPTIONS="--use_chat_template False"
fi
shopt -u nocasematch

echo "=== HELMET 128K evaluation ==="
echo "Job ID:    $SLURM_JOBID"
echo "Model:     $MODEL"
echo "Short:     $MODEL_SHORT"
echo "Options:   $OPTIONS"
echo "Configs:   ${CONFIGS[*]}"
echo "Output:    $OUTDIR"
echo "Start:     $(date)"

mkdir -p "$OUTDIR"

apptainer exec --rocm \
    -B /shared_silo/scratch:/shared_silo/scratch:rw \
    -B /usr/share/libdrm:/usr/share/libdrm:ro \
    "$IMG" bash -c "
set -euo pipefail
ulimit -n 65536

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export HF_HOME=${SCRATCH}/hf_home
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton-\${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-\${SLURM_JOB_ID}
export PYTHONUSERBASE=/tmp/pip-helmet-\${SLURM_JOB_ID}

pip install --user pytrec_eval rouge_score sentencepiece absl-py --quiet 2>&1 | tail -3

cd $HELMET_DIR

for CONFIG in ${CONFIGS[*]}; do
    echo ''
    echo '======================================='
    echo \"Running HELMET config: \$CONFIG\"
    echo '======================================='

    python eval.py \\
        --config configs/\$CONFIG \\
        --seed $SEED \\
        --output_dir $OUTDIR \\
        --tag $TAG \\
        --model_name_or_path $MODEL \\
        --use_vllm \\
        $OPTIONS

    rc=\$?
    if [ \$rc -eq 0 ]; then
        echo \"Done: \$CONFIG\"
    else
        echo \"FAILED: \$CONFIG (exit \$rc)\"
    fi
done
"

echo "End: $(date)"
echo "Results in: $OUTDIR"
