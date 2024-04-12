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

# TODO use $SLURM_JOB_ID for workdir

# All used environment variables should be checked here.
if [ -z "$MODEL" ]; then
    echo "MODEL is not set"
    exit 1
fi
if [ -z "$N_SAMPLES" ]; then
    echo "N_SAMPLES is not set"
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
if [ -z "$TRUST_REMOTE_CODE" ]; then
    echo "TRUST_REMOTE_CODE is not set"
    exit 1
fi
if [ -z "$WORK_DIR" ]; then
    echo "WORK_DIR is not set"
    exit 1
fi


# set up environment
WORK_DIR=$WORK_DIR/bigcode_eval_harness
mkdir -p $WORK_DIR
cd $WORK_DIR

mkdir -p $OUTPUT_DIR
echo Saving results to: $OUTPUT_FILE


# move cache and tmp to work dir
export PIP_CACHE_DIR=$WORK_DIR/cache
echo PIP_CACHE_DIR is $PIP_CACHE_DIR
mkdir -p $PIP_CACHE_DIR
# /tmp may not be big enough for pip downloads
OLD_TMPDIR=$TMPDIR
export TMPDIR=$WORK_DIR/tmp
mkdir -p $TMPDIR
echo TMPDIR is $TMPDIR

# setup venv
module load cray-python
python -m venv venv
source venv/bin/activate

git clone https://github.com/bigcode-project/bigcode-evaluation-harness
cd bigcode-evaluation-harness

pip install --upgrade torch --index-url https://download.pytorch.org/whl/rocm5.2
pip install transformers==4.37.2 accelerate
pip install --no-cache-dir -r requirements.txt

pip install sentencepiece --upgrade
pip install tiktoken --upgrade

# NOTE: humaneval uses a unix socket which fails with AF_UNIX path too long if
# TMPDIR is longer than 108 characters, so revert TMPDIR here.
export TMPDIR=$OLD_TMPDIR
echo TMPDIR is $TMPDIR

echo Cuda Available: "$(python -c 'import torch; print(torch.cuda.is_available())')"

python main.py \
    --model $MODEL \
    --tasks $TASK_LIST \
    --precision bf16 \
    --allow_code_execution \
    --max_memory_per_gpu=auto \
    --max_length_generation=2048 \
    --n_samples $N_SAMPLES \
    --metric_output_path $OUTPUT_FILE \
    $( [ "$TRUST_REMOTE_CODE" = "True" ] && echo "--trust_remote_code" )
