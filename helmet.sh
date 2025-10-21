#!/bin/bash
# Helper script to run HELMET long-context evaluations
# Usage: sh helmet.sh <model_path_or_hf_id> [--backend vllm] [--short]

QUEUE=standard-g
PROJECT=project_462000353

# Parse arguments
MODEL=$1
shift

BACKEND="hf"
USE_SHORT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --short)
            USE_SHORT="_short"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: sh helmet.sh <model> [--backend vllm] [--short]"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "Error: Model path or HuggingFace ID is required"
    echo "Usage: sh helmet.sh <model> [--backend vllm] [--short]"
    exit 1
fi

echo "Running HELMET evaluations for model: $MODEL"
echo "Backend: $BACKEND"
echo "Using short configs: ${USE_SHORT:-no}"
echo ""

# Run all HELMET evaluation categories
# Recall tasks (includes NIAH and RULER benchmarks)
echo "=== Running Recall Tasks ==="
python main.py --time 24:00:00 --project $PROJECT --partition $QUEUE --model $MODEL --backend $BACKEND \
    helmet_recall${USE_SHORT} helmet_niah helmet_ruler${USE_SHORT}

# RAG tasks
echo "=== Running RAG Tasks ==="
python main.py --time 24:00:00 --project $PROJECT --partition $QUEUE --model $MODEL --backend $BACKEND \
    helmet_rag${USE_SHORT}

# Reranking tasks
echo "=== Running Reranking Tasks ==="
python main.py --time 24:00:00 --project $PROJECT --partition $QUEUE --model $MODEL --backend $BACKEND \
    helmet_rerank${USE_SHORT}

# Citation tasks (includes ALCE variants)
echo "=== Running Citation Tasks ==="
python main.py --time 24:00:00 --project $PROJECT --partition $QUEUE --model $MODEL --backend $BACKEND \
    helmet_cite${USE_SHORT} helmet_alce_nocite${USE_SHORT}

# Long QA tasks
echo "=== Running Long QA Tasks ==="
python main.py --time 24:00:00 --project $PROJECT --partition $QUEUE --model $MODEL --backend $BACKEND \
    helmet_longqa${USE_SHORT}

# Summarization tasks
echo "=== Running Summarization Tasks ==="
python main.py --time 24:00:00 --project $PROJECT --partition $QUEUE --model $MODEL --backend $BACKEND \
    helmet_summ${USE_SHORT}

# In-context learning tasks
echo "=== Running ICL Tasks ==="
python main.py --time 24:00:00 --project $PROJECT --partition $QUEUE --model $MODEL --backend $BACKEND \
    helmet_icl${USE_SHORT}

echo ""
echo "All HELMET evaluations queued!"
echo "Monitor progress with: python watch.py"
