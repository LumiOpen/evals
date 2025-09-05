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
if [ -z "$WORK_DIR" ]; then
    echo "WORK_DIR is not set"
    exit 1
fi
if [ -z "$APPLY_CHAT_TEMPLATE" ]; then
    echo "APPLY_CHAT_TEMPLATE is not set"
    exit 1
fi
if [ -z "$FEWSHOT_AS_MULTITURN" ]; then
    echo "FEWSHOT_AS_MULTITURN is not set"
    exit 1
fi


# Remove any venv settings that might confuse things.
unset PYTHONPATH
unset PYTHONHOME
unset VIRTUAL_ENV

export HF_HOME="/scratch/project_462000353/hf_cache"

# Prepare work dir
WORK_DIR=$WORK_DIR/lm_eval_harness
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

# load basic python environment
module use /appl/local/csc/modulefiles
module load pytorch

### install correct packages
# Due to parallel exeuction it's possible for race conditions / collisionshere
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

# Check out correct lm-evaluation-harness code

# We have migrated repositories, this following code:
# 1. Ensures we have a copy of the right repo
# 2. Checks out the current version of the code from the repo
REPO_DIR="lm-evaluation-harness2"
REPO_URL="https://github.com/LumiOpen/lm-evaluation-harness"
BRANCH="main"

# Check if directory exists
if [ ! -d "$REPO_DIR" ] ; then
    # Directory doesn't exist, perform initial clone
    echo "Directory doesn't exist. Cloning $REPO_URL..."
    git clone -b $BRANCH $REPO_URL $REPO_DIR
else
    # Directory exists, check current remote URL
    cd $REPO_DIR
    CURRENT_REMOTE=$(git config --get remote.origin.url)
    
    if [ "$CURRENT_REMOTE" != "$REPO_URL" ] ; then
        # Different repository, backup and re-clone
        echo "Repository changed from $CURRENT_REMOTE to $REPO_URL"
        cd ..
        mv $REPO_DIR ${REPO_DIR}_backup_$(date +%Y%m%d%H%M%S)
        git clone -b $BRANCH $REPO_URL $REPO_DIR
        echo "Previous repository backed up and new one cloned"
    else
        # Same repository, just update
        echo "Updating existing repository..."
        git fetch origin
        git reset --hard origin/$BRANCH
    fi
fi
# Make sure we end up in the repository directory
if [ "$(pwd)" != *"$REPO_DIR" ]; then
    cd $REPO_DIR
fi

echo Using $REPO_URL branch $BRANCH in directory $REPO_DIR

# Setup basic python environment
export PYTHONUSERBASE=$WORK_DIR/pythonuserbase
mkdir -p $PYTHONUSERBASE
echo PYTHONUSERBASE is $PYTHONUSERBASE
export PATH=$PATH:$PYTHONUSERBASE/bin
pip install --upgrade setuptools pip
pip --python=/appl/local/csc/soft/ai/bin/python install --user  -e .[hf_transfer,math,multilingual,sentencepiece,ifeval]
pip install --upgrade langdetect immutabledict heliport

# there is an incompatibility with versions of vllm < 0.10.1 and transformers
# >= 4.54.0.  the following installs a non-functional vllm because it is not
# built for ROCm, but we don't use vllm and lm_eval_harness will continue after
# the vllm fails to load.  the startup does seem to take a long time, for
# whatever reason.
# this is a temporary workaround until we convert to running in our own
# clean singularity container, which is coming Soon.
pip install transformers==4.56.1 vllm==0.10.1.1 outlines_core==0.2.10

# remove cleanup trap and release lock
trap - EXIT
cleanup
echo "Environment setup complete."


### Prepare to run command
echo Cuda Available: "$(python -c 'import torch; print(torch.cuda.is_available())')"

# lm_eval has changed so that --output_path is treated as a directory and the
# file outputs are stored in a directory structure under that point.  we don't
# want that.  so now we output to a temporary directory then move the resulting
# file to $OUTPUT_FILE
RANDOM_DIR="/tmp/lm_eval_$(date +%s%N)"
mkdir -p "$RANDOM_DIR"
echo Saving temporary results to $RANDOM_DIR

mkdir -p $OUTPUT_DIR
echo Final results will be saved to: $OUTPUT_FILE


if [ "$APPLY_CHAT_TEMPLATE" = "True" ]; then
    CHAT_TEMPLATE_FLAG="--apply_chat_template"
else
    CHAT_TEMPLATE_FLAG=""
fi


if [ "$FEWSHOT_AS_MULTITURN" = "True" ]; then
    FEWSHOT_AS_MULTITURN_FLAG="--fewshot_as_multiturn"
else
    FEWSHOT_AS_MULTITURN_FLAG=""
fi

### Launch command

# current versions of accelerate are very picky about these environment
# variables and demand only one be set.
unset ROCR_VISIBLE_DEVICES

set -x
lm_eval \
    --model hf \
    --model_args pretrained=$MODEL,parallelize=True,tokenizer=$TOKENIZER,dtype=bfloat16,trust_remote_code=$TRUST_REMOTE_CODE,max_memory_per_gpu=60GB \
    --tasks "$TASK_LIST" \
    --num_fewshot $NUM_FEWSHOT \
    --output_path $RANDOM_DIR \
    $CHAT_TEMPLATE_FLAG \
    $FEWSHOT_AS_MULTITURN_FLAG
    # --log_samples \
set +x

echo Moving temporary results from $RANDOM_DIR to $OUTPUT_FILE
find "$RANDOM_DIR" -name "results_*.json" -exec mv {} "$OUTPUT_FILE" \;
rm -rf "$RANDOM_DIR"
