#!/usr/bin/env bash
# helmet_cpt.sh - Run all HELMET evaluations on a model with vllm backend
# Supports both HuggingFace model IDs (e.g., meta-llama/Meta-Llama-3.1-8B) and local checkpoint paths
set -euo pipefail

# Config from env, with sane defaults
PROJECT="${PROJECT:-project_462000963}"
QUEUE="${QUEUE:-standard-g}"
MODEL="${1:?usage: $0 <model_id_or_checkpoint_path>}"

# Backend is fixed to vllm for evaluation
BACKEND="vllm"

# Optional env knobs
MODEL_ARGS="${MODEL_ARGS:-}"
FORCE="${FORCE:---force}"

run() {
  local time="$1"
  local gres="$2"
  local max_len="$3"
  shift 3
  local tasks=( "$@" )

  # Build model args with context-appropriate max_model_len
  local model_args="max_model_len=$max_len"
  if [[ -n "$MODEL_ARGS" ]]; then
    model_args="$MODEL_ARGS,$model_args"
  fi

  # Build args as an array to keep quoting correct
  local args=( python main.py --backend "$BACKEND" --project "$PROJECT" --partition "$QUEUE" --model "$MODEL" --time "$time" --gres "$gres" --model_args "$model_args" )

  if [[ -n "$FORCE" ]]; then
    args+=( "$FORCE" )
  fi

  args+=( "${tasks[@]}" )

  printf '==> '; printf '%q ' "${args[@]}"; printf '\n'
  "${args[@]}"
}

echo "========================================"
echo "Running HELMET evaluations on model:"
echo "  Model: $MODEL"
echo "  Backend: $BACKEND"
echo "  Project: $PROJECT"
echo "  Queue: $QUEUE"
echo "========================================"
echo ""

# ==============================================================================
# 8k CONTEXT TASKS (max_model_len=8192)
# ==============================================================================
echo "=== Launching 8k context tasks ==="

# RAG tasks (4 datasets, per-dataset with optimized timing)
echo "--- RAG 8k (per-dataset) ---"
run 2:00:00 "gpu:mi250:2" 8192 helmet_rag_nq_8k helmet_rag_triviaqa_8k
run 1:30:00 "gpu:mi250:2" 8192 helmet_rag_hotpotqa_8k
run 0:30:00 "gpu:mi250:2" 8192 helmet_rag_popqa_8k

# Longqa tasks (3 datasets, per-dataset with optimized timing)
echo "--- Longqa 8k (per-dataset) ---"
run 1:00:00 "gpu:mi250:2" 8192 helmet_longqa_narrativeqa_8k
run 0:30:00 "gpu:mi250:2" 8192 helmet_longqa_infbench_qa_8k helmet_longqa_infbench_choice_8k

# Other tasks (bundled, unchanged)
echo "--- Other 8k tasks ---"
run 4:00:00 "gpu:mi250:2" 8192 helmet_recall_8k helmet_rerank_8k helmet_cite_8k helmet_alce_nocite_8k helmet_summ_8k helmet_icl_8k

# ==============================================================================
# 16k CONTEXT TASKS (max_model_len=16384)
# ==============================================================================
echo ""
echo "=== Launching 16k context tasks ==="

# RAG tasks (4 datasets, per-dataset with increased time limits and 8 GPUs)
echo "--- RAG 16k (per-dataset) ---"
run 3:00:00 "gpu:mi250:8" 16384 helmet_rag_nq_16k helmet_rag_triviaqa_16k
run 24:00:00 "gpu:mi250:8" 16384 helmet_rag_hotpotqa_16k
run 2:00:00 "gpu:mi250:8" 16384 helmet_rag_popqa_16k

# Longqa tasks (3 datasets, per-dataset with optimized timing)
echo "--- Longqa 16k (per-dataset) ---"
run 2:30:00 "gpu:mi250:4" 16384 helmet_longqa_narrativeqa_16k
run 0:30:00 "gpu:mi250:4" 16384 helmet_longqa_infbench_qa_16k helmet_longqa_infbench_choice_16k

# Other tasks (bundled, unchanged)
echo "--- Other 16k tasks ---"
run 12:00:00 "gpu:mi250:8" 16384 helmet_recall_16k helmet_rerank_16k helmet_cite_16k helmet_alce_nocite_16k helmet_summ_16k helmet_icl_16k

# ==============================================================================
# 32k CONTEXT TASKS (max_model_len=32768)
# ==============================================================================
echo ""
echo "=== Launching 32k context tasks ==="

# RAG tasks (4 datasets, per-dataset with optimized timing)
echo "--- RAG 32k (per-dataset) ---"
run 12:00:00 "gpu:mi250:8" 32768 helmet_rag_nq_32k helmet_rag_triviaqa_32k
run 9:00:00 "gpu:mi250:8" 32768 helmet_rag_hotpotqa_32k
run 3:00:00 "gpu:mi250:8" 32768 helmet_rag_popqa_32k

# Longqa tasks (3 datasets, per-dataset with optimized timing)
echo "--- Longqa 32k (per-dataset) ---"
run 6:00:00 "gpu:mi250:8" 32768 helmet_longqa_narrativeqa_32k
run 1:00:00 "gpu:mi250:8" 32768 helmet_longqa_infbench_qa_32k helmet_longqa_infbench_choice_32k

# Other tasks (bundled, unchanged)
echo "--- Other 32k tasks ---"
run 12:00:00 "gpu:mi250:8" 32768 helmet_recall_32k helmet_rerank_32k helmet_cite_32k helmet_alce_nocite_32k helmet_summ_32k helmet_icl_32k

# ==============================================================================
# 64k CONTEXT TASKS (max_model_len=65536)
# ==============================================================================
echo ""
echo "=== Launching 64k context tasks ==="

# RAG tasks (4 datasets, per-dataset with 24h time limit for double-tokenization overhead)
echo "--- RAG 64k (per-dataset) ---"
run 24:00:00 "gpu:mi250:8" 65536 helmet_rag_nq_64k helmet_rag_triviaqa_64k
run 24:00:00 "gpu:mi250:8" 65536 helmet_rag_hotpotqa_64k
run 24:00:00 "gpu:mi250:8" 65536 helmet_rag_popqa_64k

# Longqa tasks (3 datasets, per-dataset with optimized timing)
echo "--- Longqa 64k (per-dataset) ---"
run 13:00:00 "gpu:mi250:8" 65536 helmet_longqa_narrativeqa_64k
run 1:00:00 "gpu:mi250:8" 65536 helmet_longqa_infbench_qa_64k helmet_longqa_infbench_choice_64k

# Other tasks (bundled, unchanged)
echo "--- Other 64k tasks ---"
run 18:00:00 "gpu:mi250:8" 65536 helmet_recall_64k helmet_rerank_64k helmet_cite_64k helmet_alce_nocite_64k helmet_summ_64k helmet_icl_64k

# ==============================================================================
# 128k CONTEXT TASKS (max_model_len=131072)
# ==============================================================================
echo ""
echo "=== Launching 128k context tasks ==="

# RAG tasks (4 datasets, per-dataset with 24h time limit for double-tokenization overhead)
echo "--- RAG 128k (per-dataset) ---"
run 24:00:00 "gpu:mi250:8" 131072 helmet_rag_nq_128k helmet_rag_triviaqa_128k
run 24:00:00 "gpu:mi250:8" 131072 helmet_rag_hotpotqa_128k
run 24:00:00 "gpu:mi250:8" 131072 helmet_rag_popqa_128k

# Longqa tasks (3 datasets, per-dataset with optimized timing)
echo "--- Longqa 128k (per-dataset) ---"
run 22:00:00 "gpu:mi250:8" 131072 helmet_longqa_narrativeqa_128k
run 2:00:00 "gpu:mi250:8" 131072 helmet_longqa_infbench_qa_128k helmet_longqa_infbench_choice_128k

# Other tasks (bundled, unchanged)
echo "--- Other 128k tasks ---"
run 24:00:00 "gpu:mi250:8" 131072 helmet_recall_128k helmet_rerank_128k helmet_cite_128k helmet_alce_nocite_128k helmet_summ_128k helmet_icl_128k

echo ""
echo "========================================"
echo "All HELMET evaluation tasks queued!"
echo "  Per-dataset tasks:"
echo "    - RAG: 20 tasks (4 datasets × 5 contexts)"
echo "    - Longqa: 15 tasks (3 datasets × 5 contexts)"
echo "  Bundled tasks:"
echo "    - 6 other task types × 5 contexts = 30 tasks"
echo "  Total: 65 tasks"
echo "  Context lengths: 8k, 16k, 32k, 64k, 128k"
echo "========================================"
echo "Monitor progress with: python watch.py"
