#!/bin/bash
# Script to run RULER long context evaluations across multiple sequence lengths
# Usage: sh ruler.sh /path/to/model [--partition PARTITION] [--time TIME] [--gres GRES]

if [ -z "$1" ]; then
    echo "Usage: sh ruler.sh /path/to/model [--partition PARTITION] [--time TIME] [--gres GRES]"
    echo ""
    echo "Example:"
    echo "  sh ruler.sh /path/to/model"
    echo "  sh ruler.sh /path/to/model --partition standard-g --time 24:00:00"
    echo ""
    echo "This script will run RULER evaluations for the following sequence lengths:"
    echo "  - 4096 tokens"
    echo "  - 8192 tokens"
    echo "  - 16384 tokens"
    echo "  - 32768 tokens"
    echo "  - 65536 tokens"
    echo "  - 131072 tokens (128K)"
    exit 1
fi

MODEL=$1
shift

# Default SLURM configuration
PARTITION="standard-g"
TIME="48:00:00"
GRES="gpu:mi250:4"
BACKEND="hf"
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
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "=================================================="
echo "Running RULER Long Context Evaluations"
echo "=================================================="
echo "Model: $MODEL"
echo "Partition: $PARTITION"
echo "Time: $TIME"
echo "GRES: $GRES"
echo "Backend: $BACKEND"
echo "Extra args: $EXTRA_ARGS"
echo "=================================================="
echo ""

# RULER evaluations for each sequence length
# Note: Longer context lengths require more time and memory
RULER_TASKS=(
    "ruler_4096"
    "ruler_8192"
    "ruler_16384"
    "ruler_32768"
    "ruler_65536"
    "ruler_131072"
)

for task in "${RULER_TASKS[@]}"; do
    echo "Submitting job for $task..."
    python main.py \
        --partition "$PARTITION" \
        --time "$TIME" \
        --gres "$GRES" \
        --backend "$BACKEND" \
        --model "$MODEL" \
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

