#!/usr/bin/env bash
# helmet_cpt.sh - Run all HELMET evaluations on a checkpoint with vllm backend
set -euo pipefail

# Config from env, with sane defaults
PROJECT="${PROJECT:-project_462000963}"
QUEUE="${QUEUE:-standard-g}"
MODEL="${1:?usage: $0 <model_checkpoint_path>}"

# Backend is fixed to vllm for checkpoint evaluation
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
echo "Running HELMET evaluations on checkpoint:"
echo "  Model: $MODEL"
echo "  Backend: $BACKEND"
echo "  Project: $PROJECT"
echo "  Queue: $QUEUE"
echo "========================================"
echo ""

# 8k context tasks (2 GPUs, 4 hours, max_model_len=8192)
echo "=== Launching 8k context tasks ==="
run 4:00:00 "gpu:mi250:2" 8192 helmet_recall_8k helmet_rag_8k helmet_rerank_8k helmet_cite_8k helmet_alce_nocite_8k helmet_longqa_8k helmet_summ_8k helmet_icl_8k

# 16k context tasks (2 GPUs, 6 hours, max_model_len=16384)
echo "=== Launching 16k context tasks ==="
run 6:00:00 "gpu:mi250:2" 16384 helmet_recall_16k helmet_rag_16k helmet_rerank_16k helmet_cite_16k helmet_alce_nocite_16k helmet_longqa_16k helmet_summ_16k helmet_icl_16k

# 32k context tasks (4 GPUs, 8 hours, max_model_len=32768)
echo "=== Launching 32k context tasks ==="
run 8:00:00 "gpu:mi250:4" 32768 helmet_recall_32k helmet_rag_32k helmet_rerank_32k helmet_cite_32k helmet_alce_nocite_32k helmet_longqa_32k helmet_summ_32k helmet_icl_32k

# 64k context tasks (4 GPUs, 12 hours, max_model_len=65536)
echo "=== Launching 64k context tasks ==="
run 12:00:00 "gpu:mi250:4" 65536 helmet_recall_64k helmet_rag_64k helmet_rerank_64k helmet_cite_64k helmet_alce_nocite_64k helmet_longqa_64k helmet_summ_64k helmet_icl_64k

# 128k context tasks (8 GPUs, 24 hours, max_model_len=131072)
echo "=== Launching 128k context tasks ==="
run 24:00:00 "gpu:mi250:8" 131072 helmet_recall_128k helmet_rag_128k helmet_rerank_128k helmet_cite_128k helmet_alce_nocite_128k helmet_longqa_128k helmet_summ_128k helmet_icl_128k

echo ""
echo "========================================"
echo "All 40 HELMET evaluation tasks queued!"
echo "  - 8 tasks × 5 context lengths"
echo "  - 8k, 16k, 32k, 64k, 128k"
echo "  - recall, rag, rerank, cite, alce_nocite, longqa, summ, icl"
echo "========================================"
echo "Monitor progress with: python watch.py"
