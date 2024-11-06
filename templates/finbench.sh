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
WORK_DIR=$WORK_DIR/finbench
mkdir -p $WORK_DIR
cd $WORK_DIR

mkdir -p $OUTPUT_DIR
echo Saving results to: $OUTPUT_FILE


# move cache and tmp to work dir
export PIP_CACHE_DIR=$WORK_DIR/cache
echo PIP_CACHE_DIR is $PIP_CACHE_DIR
mkdir -p $PIP_CACHE_DIR
# /tmp may not be big enough for pip downloads
export TMPDIR=$WORK_DIR/tmp
mkdir -p $TMPDIR
echo TMPDIR is $TMPDIR

# setup venv
ml use /appl/local/csc/modulefiles/
ml pytorch/2.2
python -m venv --system-site-packages venv
source venv/bin/activate

git clone -b finnish https://github.com/jonabur/lm-evaluation-harness.git
cd lm-evaluation-harness

#pip install --upgrade torch --index-url https://download.pytorch.org/whl/rocm5.6
pip install transformers==4.37.2
pip install --no-cache-dir -r requirements.txt

pip install datasets==2.20.0
export HF_DATASETS_TRUST_REMOTE_CODE=TRUE
export PYTHONPATH="${PYTHONPATH}:/scratch/project_462000353/akselir/.local/bin"
echo Cuda Available: "$(python -c 'import torch; print(f"{torch.cuda.is_available()},version {torch.__version__}")')"
device="$(python -c 'import torch; d=torch.device("cuda" if torch.cuda.is_available() else "cpu");print(d)')"

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL,use_accelerate=True,tokenizer=$TOKENIZER,dtype=bfloat16,trust_remote_code=$TRUST_REMOTE_CODE \
    --device $device \
    --no_cache \
    --tasks bigbench_1_digit_addition,bigbench_1_digit_division,bigbench_1_digit_multiplication,bigbench_1_digit_subtraction,bigbench_2_digit_addition,bigbench_2_digit_division,bigbench_2_digit_multiplication,bigbench_2_digit_subtraction,bigbench_3_digit_addition,bigbench_3_digit_division,bigbench_3_digit_multiplication,bigbench_3_digit_subtraction,bigbench_4_digit_addition,bigbench_4_digit_division,bigbench_4_digit_multiplication,bigbench_4_digit_subtraction,bigbench_5_digit_addition,bigbench_5_digit_division,bigbench_5_digit_multiplication,bigbench_5_digit_subtraction,bigbench_analogies,bigbench_emotions,bigbench_empirical_judgments,bigbench_general_knowledge,bigbench_harmless,bigbench_helpful,bigbench_honest,bigbench_intent_recognition,bigbench_misconceptions,bigbench_one_sentence,bigbench_one_sentence_no_prompt,bigbench_other,bigbench_paraphrase,bigbench_sentence_ambiguity,bigbench_similarities_abstraction,bigbench_two_sentences \
    --num_fewshot $NUM_FEWSHOT \
    --output_path $OUTPUT_FILE
