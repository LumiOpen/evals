#!/bin/bash
# Quick HELMET evaluation script using demo/short configs for faster testing
# Usage: sh helmet-quick.sh <model_path_or_hf_id> [--backend vllm]

QUEUE=standard-g
PROJECT=project_462000353

# Parse arguments
MODEL=$1
shift

BACKEND="hf"

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sh helmet-quick.sh <model> [--backend vllm]"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "Error: Model path or HuggingFace ID is required"
    echo "Usage: sh helmet-quick.sh <model> [--backend vllm]"
    exit 1
fi

echo "Running quick HELMET evaluations for model: $MODEL"
echo "Backend: $BACKEND"
echo ""

# Run a subset of quick HELMET evaluations for testing
echo "=== Running Quick HELMET Tests ==="
python main.py --time 04:00:00 --project $PROJECT --partition $QUEUE --model $MODEL --backend $BACKEND \
    helmet_recall_demo helmet_recall_short helmet_rag_short

echo ""
echo "Quick HELMET evaluations queued!"
echo "Monitor progress with: python watch.py"
