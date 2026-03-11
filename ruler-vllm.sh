#!/bin/bash
# Script to run RULER long context evaluations using vLLM backend
# Each subtask × sequence length combination runs as a separate SLURM job
#
# Usage: sh ruler-vllm.sh /path/to/model [OPTIONS]

if [ -z "$1" ]; then
    cat <<EOF
Usage: sh ruler-vllm.sh /path/to/model [OPTIONS]

Examples:
  # Run all RULER subtasks at all sequence lengths (78 jobs)
  sh ruler-vllm.sh org/modelname

  # Run specific subtasks at all sequence lengths
  sh ruler-vllm.sh /path/to/model --subtasks "niah_single_1,ruler_vt,ruler_qa_hotpot"

  # Run all subtasks at specific sequence lengths
  sh ruler-vllm.sh /path/to/model --sequence-lengths "4096,8192,16384"

  # Run specific combinations
  sh ruler-vllm.sh /path/to/model --subtasks "niah_single_1,ruler_vt" --sequence-lengths "4096,8192"

  # Custom SLURM configuration
  sh ruler-vllm.sh /path/to/model --partition standard-g --time 24:00:00 --gres gpu:mi250:8

  # SFT/instruct model with chat template
  sh ruler-vllm.sh /path/to/sft-model --apply_chat_template --fewshot_as_multiturn

  # SFT model via environment variables (same as cpt-vllm.sh pattern)
  APPLY_CHAT_TEMPLATE=True FEWSHOT_AS_MULTITURN=True sh ruler-vllm.sh /path/to/sft-model

Available RULER subtasks (13 total):
  Needle in a Haystack (NIAH):
    niah_single_1, niah_single_2, niah_single_3
    niah_multikey_1, niah_multikey_2, niah_multikey_3
    niah_multivalue, niah_multiquery
  Other tasks:
    ruler_vt (Variable Tracking)
    ruler_cwe (Common Words Extraction)
    ruler_fwe (Frequent Words Extraction)
    ruler_qa_hotpot (QA - Hotpot)
    ruler_qa_squad (QA - SQuADv2)

Available sequence lengths:
  4096, 8192, 16384, 32768, 65536, 131072

Note: vLLM backend is faster but experimental.
      Each subtask × sequence length combination runs as a separate SLURM job.
EOF
    exit 1
fi

MODEL=$1
shift

# Default configuration
PARTITION="standard-g"
TIME="48:00:00"
GRES="gpu:mi250:4"
EXTRA_ARGS=""

# Chat template flags (also read from env vars, same pattern as cpt-vllm.sh)
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-}"
FEWSHOT_AS_MULTITURN="${FEWSHOT_AS_MULTITURN:-}"

# Default: all subtasks and all sequence lengths
# From: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/ruler#tasks
SUBTASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_hotpot,ruler_qa_squad"
SEQUENCE_LENGTHS="4096,8192,16384,32768,65536,131072"

# Parse arguments
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
        --subtasks)
            SUBTASKS="$2"
            shift 2
            ;;
        --sequence-lengths)
            SEQUENCE_LENGTHS="$2"
            shift 2
            ;;
        --apply_chat_template|--apply-chat-template)
            APPLY_CHAT_TEMPLATE="True"
            shift
            ;;
        --fewshot_as_multiturn|--fewshot-as-multiturn)
            FEWSHOT_AS_MULTITURN="True"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Convert comma-separated strings to arrays
IFS=',' read -ra SUBTASK_ARRAY <<< "$SUBTASKS"
IFS=',' read -ra SEQLEN_ARRAY <<< "$SEQUENCE_LENGTHS"

# Count total jobs
TOTAL_JOBS=$((${#SUBTASK_ARRAY[@]} * ${#SEQLEN_ARRAY[@]}))

# Map sequence lengths to max_model_len (with buffer)
declare -A MAX_MODEL_LEN_MAP
MAX_MODEL_LEN_MAP[4096]=8192
MAX_MODEL_LEN_MAP[8192]=16384
MAX_MODEL_LEN_MAP[16384]=32768
MAX_MODEL_LEN_MAP[32768]=65536
MAX_MODEL_LEN_MAP[65536]=131072
MAX_MODEL_LEN_MAP[131072]=131072

echo "=================================================="
echo "Running RULER Long Context Evaluations (vLLM)"
echo "=================================================="
echo "Model: $MODEL"
echo "Partition: $PARTITION"
echo "Time: $TIME"
echo "GRES: $GRES"
echo "Backend: vllm"
echo "Apply chat template: ${APPLY_CHAT_TEMPLATE:-no}"
echo "Fewshot as multiturn: ${FEWSHOT_AS_MULTITURN:-no}"
echo "Extra args: $EXTRA_ARGS"
echo ""
echo "Subtasks: ${#SUBTASK_ARRAY[@]} (${SUBTASKS})"
echo "Sequence lengths: ${#SEQLEN_ARRAY[@]} (${SEQUENCE_LENGTHS})"
echo "Total SLURM jobs to submit: $TOTAL_JOBS"
echo "=================================================="
echo ""

# Confirm if submitting many jobs
if [ $TOTAL_JOBS -gt 20 ]; then
    echo "WARNING: You are about to submit $TOTAL_JOBS SLURM jobs."
    echo "Press Ctrl+C to cancel, or Enter to continue..."
    read -r
fi

JOB_COUNT=0

# Submit jobs for each combination
for subtask in "${SUBTASK_ARRAY[@]}"; do
    for seqlen in "${SEQLEN_ARRAY[@]}"; do
        JOB_COUNT=$((JOB_COUNT + 1))
        TASK_NAME="ruler_${subtask}_${seqlen}"
        MAX_LEN="${MAX_MODEL_LEN_MAP[$seqlen]}"
        
        echo "[$JOB_COUNT/$TOTAL_JOBS] Submitting SLURM job for $TASK_NAME (max_model_len=$MAX_LEN)..."
        
        # This calls main.py which internally generates and submits a SLURM job via sbatch
        python3 main.py \
            --partition "$PARTITION" \
            --time "$TIME" \
            --gres "$GRES" \
            --backend vllm \
            --model "$MODEL" \
            --model_args "max_model_len=$MAX_LEN,gpu_memory_utilization=0.95" \
            ${APPLY_CHAT_TEMPLATE:+--apply_chat_template} \
            ${FEWSHOT_AS_MULTITURN:+--fewshot_as_multiturn} \
            $EXTRA_ARGS \
            -- \
            "$TASK_NAME"
        
        # Brief pause to avoid overwhelming the scheduler
        sleep 0.5
    done
done

echo ""
echo "=================================================="
echo "All $TOTAL_JOBS RULER evaluation jobs submitted!"
echo "=================================================="
echo ""
echo "Each job runs independently in SLURM."
echo ""
echo "Monitor job status with:"
echo "  python watch.py"
echo ""
echo "View results when complete with:"
echo "  sh summary.sh output/v2/<model-name>"
echo ""
echo "Check specific task results:"
echo "  ls output/v2/<model-name>/ruler_*"
echo ""
