#!/bin/bash
#SBATCH --job-name={{ slurm_config.name }}
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --cpus-per-task=32

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
if [ -z "$NUM_FEWSHOT" ]; then
    echo "NUM_FEWSHOT is not set"
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
if [ -z "$TASK_LIST" ]; then
    echo "TASK_LIST is not set"
    exit 1
fi
if [ -z "$TOKENIZER" ]; then
    echo "TOKENIZER is not set"
    exit 1
fi
if [ -z "$TRUST_REMOTE_CODE" ]; then
    echo "TRUST_REMOTE_CODE is not set"
    exit 1
fi
if [ -z "$APPLY_CHAT_TEMPLATE" ]; then
    echo "APPLY_CHAT_TEMPLATE is not set"
    exit 1
fi
if [ -z "$WORK_DIR" ]; then
    echo "WORK_DIR is not set"
    exit 1
fi

# set up environment
WORK_DIR=$WORK_DIR/lm_eval_harness
mkdir -p $WORK_DIR
cd $WORK_DIR

mkdir -p $OUTPUT_DIR
echo Saving results to: $OUTPUT_FILE


# move cache and tmp to work dir so we don't run out of space
export PIP_CACHE_DIR=$WORK_DIR/cache
echo PIP_CACHE_DIR is $PIP_CACHE_DIR
mkdir -p $PIP_CACHE_DIR
export TMPDIR=$WORK_DIR/tmp
mkdir -p $TMPDIR
echo TMPDIR is $TMPDIR

module load cray-python

### setup venv

LOCKFILE="${WORK_DIR}/.setup_lock_file"
CLEANUP_LOCK="${LOCKFILE}.cleanup"

cleanup() {
    # Only remove the lock file if we own it
    if [ -f "$LOCKFILE" ] && [ "$(cat "$LOCKFILE")" = "$SLURM_JOB_ID" ]; then
        rm -f "$LOCKFILE"
    fi
    if [ -f "$CLEANUP_LOCK" ] && [ "$(cat "$CLEANUP_LOCK")" = "$SLURM_JOB_ID" ]; then
        rm -f "$CLEANUP_LOCK"
    fi
}

echo Acquiring lock for environment update: $LOCKFILE
# Try to acquire the lock for up to 1 hour (3600 seconds)
# Modify timeout as needed for your use case
TIMEOUT=3600
ATTEMPTS=0
    
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

### lock acquired
# Register the cleanup function to run on script exit
trap cleanup EXIT

echo "Acquired lock (Job ID: $SLURM_JOB_ID), executing environment setup code."
    
# run protected code
python -m venv venv.lm-evaluation-harness2
source venv.lm-evaluation-harness2/bin/activate

if [ ! -d lm-evaluation-harness2 ] ; then
   	git clone -b main https://github.com/jonabur/lm-evaluation-harness lm-evaluation-harness2
fi
cd lm-evaluation-harness2
git fetch origin
git reset --hard origin/main

pip install --upgrade torch --index-url https://download.pytorch.org/whl/rocm6.0
pip install --upgrade transformers accelerate hf_transfer
pip install --no-cache-dir -r requirements.txt
pip install jinja2 --upgrade  # missing requirement?

pip install sentencepiece --upgrade  # required for 01-ai/Yi
pip install protobuf --upgrade       # required for Llama 2 tokenizer
pip install tiktoken --upgrade

pip install -e .
pip install -e .[math]

# remove cleanup trap and release lock
trap - EXIT
cleanup

echo "Environment setup complete."



echo Cuda Available: "$(python -c 'import torch; print(torch.cuda.is_available())')"


# lm_eval has changed so that --output_path is treated as a directory and the
# file outputs are stored in a directory structure under that point.  we don't
# want that.  so now we output to a temporary directory then move the resulting
# file to $OUTPUT_FILE
RANDOM_DIR="/tmp/lm_eval_$(date +%s%N)"
mkdir -p "$RANDOM_DIR"
echo Saving temporary results to $RANDOM_DIR


if [ "$APPLY_CHAT_TEMPLATE" = "False" ]; then
    CHAT_TEMPLATE_FLAG=""
elif [ "$APPLY_CHAT_TEMPLATE" = "True" ]; then
    CHAT_TEMPLATE_FLAG="--apply_chat_template"
else
    CHAT_TEMPLATE_FLAG="--apply_chat_template ${APPLY_CHAT_TEMPLATE}"
fi

set -x
lm_eval \
    --model hf \
    --model_args pretrained=$MODEL,parallelize=True,tokenizer=$TOKENIZER,dtype=bfloat16,trust_remote_code=$TRUST_REMOTE_CODE,max_memory_per_gpu=60GB \
    --tasks "$TASK_LIST" \
    --num_fewshot $NUM_FEWSHOT \
    --output_path $RANDOM_DIR \
    $CHAT_TEMPLATE_FLAG
    # --log_samples 
set +x

echo Moving temporary results from $RANDOM_DIR to $OUTPUT_FILE
find "$RANDOM_DIR" -name "results_*.json" -exec mv {} "$OUTPUT_FILE" \;
rm -rf "$RANDOM_DIR"
