#!/bin/bash
# Script to run RULER long context evaluations using vLLM backend
# Usage: sh ruler-vllm.sh /path/to/model [--partition PARTITION] [--time TIME] [--gres GRES]

if [ -z "$1" ]; then
    echo "Usage: sh ruler-vllm.sh /path/to/model [--partition PARTITION] [--time TIME] [--gres GRES]"
    echo ""
    echo "Example:"
    echo "  sh ruler-vllm.sh org/modelname"
    echo "  sh ruler-vllm.sh /path/to/model --partition standard-g --time 24:00:00"
    echo ""
    echo "This script will run RULER evaluations using vLLM backend for the following sequence lengths:"
    echo "  - 4096 tokens"
    echo "  - 8192 tokens"
    echo "  - 16384 tokens"
    echo "  - 32768 tokens"
    echo "  - 65536 tokens"
    echo "  - 131072 tokens (128K)"
    echo ""
    echo "Note: vLLM backend is faster but experimental."
    exit 1
fi

MODEL=$1
shift

# Default SLURM configuration
# vLLM can be more efficient with memory, so we can use smaller resources
PARTITION="standard-g"
TIME="48:00:00"
GRES="gpu:mi250:4"
EXTRA_ARGS=""

# Parse additional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --gres)
            GRES="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "=================================================="
echo "Running RULER Long Context Evaluations (vLLM)"
echo "=================================================="
echo "Model: $MODEL"
echo "Partition: $PARTITION"
echo "Time: $TIME"
echo "GRES: $GRES"
echo "Backend: vllm"
echo "Extra args: $EXTRA_ARGS"
echo "=================================================="
echo ""

# RULER evaluations for each sequence length
# Note: vLLM requires max_model_len to be set appropriately for each sequence length
RULER_TASKS=(
    "ruler_4096"
    "ruler_8192"
    "ruler_16384"
    "ruler_32768"
    "ruler_65536"
    "ruler_131072"
)

# Corresponding max_model_len values (add some buffer for the context)
MAX_MODEL_LENS=(
    "8192"
    "16384"
    "32768"
    "65536"
    "131072"
    "131072"
)

for i in "${!RULER_TASKS[@]}"; do
    task="${RULER_TASKS[$i]}"
    max_len="${MAX_MODEL_LENS[$i]}"
    
    echo "Submitting job for $task (max_model_len=$max_len)..."
    python main.py \
        --partition "$PARTITION" \
        --time "$TIME" \
        --gres "$GRES" \
        --backend vllm \
        --model "$MODEL" \
        --model_args "max_model_len=$max_len,gpu_memory_utilization=0.95" \
        $EXTRA_ARGS \
        "$task"
    echo ""
done

echo "=================================================="
echo "All RULER evaluation jobs submitted!"
echo "=================================================="
echo ""
echo "Monitor job status with:"
echo "  python watch.py"
echo ""
echo "View results when complete with:"
echo "  sh summary.sh output/v2/<model-name>"

