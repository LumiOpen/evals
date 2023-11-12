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

# setup venv
module load cray-python
python -m venv venv
source venv/bin/activate

git clone https://github.com/jonabur/lm-evaluation-harness.git
cd lm-evaluation-harness

pip install --upgrade torch --index-url https://download.pytorch.org/whl/rocm5.2
pip install --no-cache-dir -r requirements.txt

# required for 01-ai/Yi
pip install sentencepiece

echo Cuda Available: "$(python -c 'import torch; print(torch.cuda.is_available())')"

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL,use_accelerate=True,tokenizer=$TOKENIZER,dtype=bfloat16,trust_remote_code=$TRUST_REMOTE_CODE \
    --device cuda:0 \
    --no_cache \
    --tasks "$TASK_LIST" \
    --num_fewshot $NUM_FEWSHOT \
    --output_path $OUTPUT_FILE
