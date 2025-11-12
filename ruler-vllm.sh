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
  sh ruler-vllm.sh /path/to/model --subtasks "niah_single_1,vt,qa_1"

  # Run all subtasks at specific sequence lengths
  sh ruler-vllm.sh /path/to/model --sequence-lengths "4096,8192,16384"

  # Run specific combinations
  sh ruler-vllm.sh /path/to/model --subtasks "niah_single_1,vt" --sequence-lengths "4096,8192"

  # Custom SLURM configuration
  sh ruler-vllm.sh /path/to/model --partition standard-g --time 24:00:00 --gres gpu:mi250:8

Available RULER subtasks (13 total):
  Needle in a Haystack (NIAH):
    niah_single_1, niah_single_2, niah_single_3
    niah_multikey_1, niah_multikey_2, niah_multikey_3
    niah_multivalue, niah_multiquery
  Other tasks:
    vt (Variable Tracking)
    cwe (Common Words Extraction)
    fwe (Frequent Words Extraction)
    qa_1, qa_2 (Question Answering)

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

# Default: all subtasks and all sequence lengths
SUBTASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2"
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
            $EXTRA_ARGS \
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
