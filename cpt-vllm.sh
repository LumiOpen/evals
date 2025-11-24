#!/usr/bin/env bash
# cpt-vllm.sh
set -euo pipefail

# Config from env, with sane defaults
PROJECT="${PROJECT:-project_462000963}"
QUEUE="${QUEUE:-standard-g}"
MODEL="${1:?usage: $0 <model_id_or_path>}"

# Backend is fixed to vllm here
BACKEND="vllm"

# Optional env knobs
# Set CPT_GRES to pass a specific gres, else do not pass --gres
CPT_GRES="${CPT_GRES:-}"
# Extra model args, for example:
# MODEL_ARGS="max_model_len=6144,gpu_memory_utilization=0.93,max_num_batched_tokens=12288"
MODEL_ARGS="${MODEL_ARGS:-}"
# Chat flags
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-}"
FEWSHOT_AS_MULTITURN="${FEWSHOT_AS_MULTITURN:-}"

run() {
  local t="$1"; shift
  local tasks=( "$@" )

  # Build args as an array to keep quoting correct
  local args=( python3 main.py --backend "$BACKEND" --project "$PROJECT" --partition "$QUEUE" --model "$MODEL" --time "$t" )

  # Only pass --gres if explicitly provided
  if [[ -n "$CPT_GRES" ]]; then
    args+=( --gres "$CPT_GRES" )
  fi

  if [[ -n "$MODEL_ARGS" ]]; then
    args+=( --model_args "$MODEL_ARGS" )
  fi
  if [[ -n "$APPLY_CHAT_TEMPLATE" ]]; then
    args+=( --apply_chat_template "$APPLY_CHAT_TEMPLATE" )
  fi
  if [[ -n "$FEWSHOT_AS_MULTITURN" ]]; then
    args+=( --fewshot_as_multiturn "$FEWSHOT_AS_MULTITURN" )
  fi

  args+=( "${tasks[@]}" )

  printf '==> '; printf '%q ' "${args[@]}"; printf '\n'
  "${args[@]}"
}

# Suites
run 12:00:00 flores200_trans_en_fi flores200_trans_fi_en
run 10:00:00 arc_challenge arc_challenge_mt_fi truthfulqa_mc truthfulqa_mc_mt_fi goldenswag_mt_fi
run 10:00:00 mmlu mmlu_mt_fi
run 48:00:00 gsm8k gsm8k_mt_fi hellaswag hellaswag_mt_fi goldenswag

