#!/bin/bash
#SBATCH --job-name={{ slurm_config.name }}
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

#SBATCH --output={{ slurm_config.log_dir }}/%j.out
#SBATCH --error={{ slurm_config.log_dir }}/%j.err

#SBATCH --account={{ slurm_config.account }}
#SBATCH --partition={{ slurm_config.partition }}
#SBATCH --gres={{ slurm_config.gres }}
#SBATCH --time={{ slurm_config.time }}

# link latest log files
ln -sf {{ slurm_config.log_dir }}/$SLURM_JOB_ID.out {{ slurm_config.log_dir }}/latest.out
ln -sf {{ slurm_config.log_dir }}/$SLURM_JOB_ID.err {{ slurm_config.log_dir }}/latest.err

# load env vars
{% for key, value in env_vars.items() %}
export {{ key }}={{ value }}
{% endfor %}

# All used environment variables should be checked here.
if [ -z "$MODEL" ]; then
    echo "MODEL is not set"
    exit 1
fi
if [ -z "$OUTPUT_DIR" ]; then
    echo "OUTPUT_DIR is not set"
    exit 1
fi
if [ -z "$OUTPUT_FILE" ]; then
    echo "OUTPUT_FILE is not set"
    exit 1
fi
if [ -z "$WORK_DIR" ]; then
    echo "WORK_DIR is not set"
    exit 1
fi
if [ -z "$INPUT_DATA_CSV" ]; then
    echo "INPUT_DATA_CSV is not set"
    exit 1
fi

# Set defaults for optional vars
BATCH_SIZE=${BATCH_SIZE:-64}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}

# Remove any venv settings that might confuse things.
unset PYTHONPATH
unset PYTHONHOME
unset VIRTUAL_ENV

export HF_HOME="/scratch/{{ slurm_config.account }}/cache/huggingface"

# Prepare work dir
WORK_DIR=$WORK_DIR/multi_if
echo WORK_DIR is $WORK_DIR
mkdir -p $WORK_DIR
cd $WORK_DIR

# move cache and tmp to work dir so we don't run out of space
export PIP_CACHE_DIR=$WORK_DIR/cache
mkdir -p $PIP_CACHE_DIR
echo PIP_CACHE_DIR is $PIP_CACHE_DIR
export TMPDIR=$WORK_DIR/tmp
mkdir -p $TMPDIR
echo TMPDIR is $TMPDIR

# Load Python environment with PyTorch
module use /appl/local/csc/modulefiles
module load pytorch/2.5

### install correct packages
# Due to parallel execution it's possible for race conditions / collisions here
# So we lock this section
LOCKFILE="${WORK_DIR}/.setup_lock_file"
CLEANUP_LOCK="${LOCKFILE}.cleanup"

# function to cleanup locks on exit
cleanup() {
    # Only remove the lock file if we own it
    if [ -f "$LOCKFILE" ] && [ "$(cat "$LOCKFILE")" = "$SLURM_JOB_ID" ]; then
        rm -f "$LOCKFILE"
    fi
    if [ -f "$CLEANUP_LOCK" ] && [ "$(cat "$CLEANUP_LOCK")" = "$SLURM_JOB_ID" ]; then
        rm -f "$CLEANUP_LOCK"
    fi
}

# Try to acquire the lock for up to 1 hour (3600 seconds)
# Modify timeout as needed for your use case
TIMEOUT=3600
ATTEMPTS=0
echo Acquiring lock for environment update: $LOCKFILE

until (set -o noclobber; echo "$SLURM_JOB_ID" > "$LOCKFILE") 2>/dev/null; do
    if [ $ATTEMPTS -ge $TIMEOUT ]; then
        echo "Failed to acquire lock after ${TIMEOUT} seconds. Exiting."
        exit 1
    fi
    
    # Check if the lock holder is still running
    if [ -f "$LOCKFILE" ]; then
        LOCK_JOBID=$(cat "$LOCKFILE")
        if ! squeue -j "$LOCK_JOBID" &>/dev/null; then
            # Lock holder is no longer running, attempt to acquire cleanup lock.
            # this it to avoid a race during cleanup when multiple jobs are waiting.
            if (set -o noclobber; echo "$SLURM_JOB_ID" > "$CLEANUP_LOCK") 2>/dev/null ; then
                rm -f "$LOCKFILE"
                rm -f "$CLEANUP_LOCK"
            fi
            # if we didn't get the lock, another process will clean it up.
        fi
    fi
        
    ATTEMPTS=$((ATTEMPTS + 1))
    sleep 1
done

### Lock acquired
# Register the cleanup function to run on script exit
trap cleanup EXIT
echo "Acquired lock (Job ID: $SLURM_JOB_ID), executing environment setup code."

### Begin protected code

# Setup user environment
export PYTHONUSERBASE=$WORK_DIR/pythonuserbase
mkdir -p $PYTHONUSERBASE
export PATH=$PATH:$PYTHONUSERBASE/bin

# Clone/update Multi-IF repository
if [ ! -d "Multi-IF" ]; then
    echo "Cloning Multi-IF repository..."
    git clone https://github.com/LumiOpen/Multi-IF.git
else
    echo "Updating Multi-IF repository..."
    cd Multi-IF
    git fetch origin
    git reset --hard origin/main
    cd ..
fi

cd Multi-IF
pip install --upgrade --user setuptools pip
pip install --user -r requirements.txt

# remove cleanup trap and release lock
trap - EXIT
cleanup
echo "Environment setup complete."

### Prepare to run command
echo Cuda Available: "$(python -c 'import torch; print(torch.cuda.is_available())')"

MODEL_ID=$(basename "$MODEL")
echo "Running Multi-IF evaluation for model: $MODEL"
echo "Model ID: $MODEL_ID"
echo "Batch size: $BATCH_SIZE"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"

mkdir -p $OUTPUT_DIR

### Launch command
set -x
python multi_turn_instruct_following_eval_vllm.py \
    --model_path "$MODEL" \
    --tokenizer_path "$MODEL" \
    --input_data_csv "$INPUT_DATA_CSV" \
    --batch_size "$BATCH_SIZE" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"

# Generate results summary
echo "Generating results summary for $MODEL_ID..."
result_output=$(sh show_result.sh "$MODEL_ID" 2>/dev/null)

# Display the output for debugging
echo "$result_output"

# Parse the output to extract scores
overall_avg=$(echo "$result_output" | grep "Multi-IF overall average:" | sed 's/Multi-IF overall average: \([0-9.]*\)/\1/')
english_avg=$(echo "$result_output" | grep "Multi-IF English average:" | sed 's/Multi-IF English average: \([0-9.]*\)/\1/')

# Handle cases where scores might be empty (set defaults)
overall_avg=${overall_avg:-0.00}
english_avg=${english_avg:-0.00}

# Create JSON output
cat > "$OUTPUT_FILE" << EOF
{
  "config": {
    "input_data_csv": "$INPUT_DATA_CSV"
  },
  "results": {
    "overall_average": $overall_avg,
    "english_average": $english_avg
  }
}
EOF

echo "Results saved to $OUTPUT_FILE:"

set +x
