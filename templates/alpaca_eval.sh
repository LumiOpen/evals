#!/bin/bash
#SBATCH --job-name={{ slurm_config.name }}
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

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
if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY is not set"
    exit 1
fi
if [ -z "$OPENAI_ORG_ID" ]; then
    echo "OPENAI_ORG_ID is not set"
    exit 1
fi
if [ -z "$RESULTS_DIR" ]; then
    echo "RESULTS_DIR is not set"
    exit 1
fi

# Set defaults for optional vars
LANGUAGE=${LANGUAGE:-"en"}
ANNOTATOR_CONFIG=${ANNOTATOR_CONFIG:-"weighted_alpaca_eval_gpt-4o-2024-08-06"}

# Remove any venv settings that might confuse things.
unset PYTHONPATH
unset PYTHONHOME
unset VIRTUAL_ENV

export HF_HOME="/scratch/project_462000353/hf_cache"

# Prepare work dir
WORK_DIR=$WORK_DIR/alpaca_eval
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

# Setup user environment
export PYTHONUSERBASE=$WORK_DIR/pythonuserbase
mkdir -p $PYTHONUSERBASE
export PATH=$PATH:$PYTHONUSERBASE/bin

# Clone/update alpaca_eval repository
if [ ! -d "alpaca_eval" ]; then
    echo "Cloning alpaca_eval repository..."
    git clone https://github.com/LumiOpen/alpaca_eval.git
else
    echo "Updating alpaca_eval repository..."
    cd alpaca_eval
    git fetch origin
    git reset --hard origin/main
    cd ..
fi

cd alpaca_eval

# Create OpenAI config
mkdir -p client_configs
cat > client_configs/openai_configs.yaml << EOF
default:
  - api_key: "${OPENAI_API_KEY}"
    organization: "${OPENAI_ORG_ID}"
EOF

echo "Created the OpenAI client config at client_configs/openai_configs.yaml"

pip install --upgrade --user setuptools pip
pip install --user -e .
pip install --user fasttext

# Setup model configuration
MODEL_ID=$(basename "$MODEL")
echo "Running AlpacaEval for model: $MODEL_ID"

# Create simple model configuration
mkdir -p "src/alpaca_eval/models_configs/$MODEL_ID"

# Create model configuration for the current model
config_file="src/alpaca_eval/models_configs/$MODEL_ID/configs.yaml"
prompt_file="src/alpaca_eval/models_configs/$MODEL_ID/prompt.txt"

cat <<EOF > "$config_file"
$MODEL_ID:
  prompt_template: "$MODEL_ID/prompt.txt"
  fn_completions: "vllm_local_completions"
  completions_kwargs:
    model_name: "$MODEL"
    model_kwargs:
      max_model_len: 4096
      tensor_parallel_size: 1
    max_new_tokens: 2048
    temperature: 0.6
    top_p: 0.9
    batch_size: 64
  pretty_name: "$MODEL_ID"
  link: ""
EOF

echo "Created YAML config at: $config_file"

 # Create prompt template
cat <<EOF > "$prompt_file"
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
EOF

echo "Created prompt template at: $prompt_file"

### Prepare to run command
echo "Running AlpacaEval for model: $MODEL"
echo "Model ID: $MODEL_ID"
echo "Language: $LANGUAGE"
echo "Annotator: $ANNOTATOR_CONFIG"

mkdir -p $OUTPUT_DIR

export SSL_CERT_FILE=$(python -m certifi)

echo "Intermediate results in: $RESULTS_DIR"

### Launch command
set -x
python -c "
import alpaca_eval
alpaca_eval.evaluate_from_model(
    model_configs='$MODEL_ID',
    output_path='$RESULTS_DIR',
    annotators_config='$ANNOTATOR_CONFIG'
)
"

# Extract results from the saved leaderboard
echo "Extracting results from leaderboard..."
LEADERBOARD_FILE="$RESULTS_DIR/$ANNOTATOR_CONFIG/leaderboard.csv"

if [ ! -f "$LEADERBOARD_FILE" ]; then
    echo "Error: Leaderboard file not found at $LEADERBOARD_FILE"
    exit 1
fi

# Parse the CSV to extract metrics for our model
MODEL_ROW=$(awk -F',' -v model="$MODEL_ID" '$1 == model' "$LEADERBOARD_FILE")

if [ -z "$MODEL_ROW" ]; then
    echo "Error: Model $MODEL_ID not found in leaderboard"
    echo "Available models:"
    awk -F',' 'NR>1 {print $1}' "$LEADERBOARD_FILE" 2>/dev/null
    exit 1
fi

# Extract values using correct column indices based on the CSV structure:
# Columns: ,win_rate,standard_error,mode,avg_length,n_wins,n_wins_base,n_draws,n_total,discrete_win_rate,length_controlled_winrate,lc_standard_error
win_rate=$(echo "$MODEL_ROW" | awk -F',' '{print $2}')
standard_error=$(echo "$MODEL_ROW" | awk -F',' '{print $3}')
avg_length=$(echo "$MODEL_ROW" | awk -F',' '{print $5}')
n_wins=$(echo "$MODEL_ROW" | awk -F',' '{print $6}')
n_wins_base=$(echo "$MODEL_ROW" | awk -F',' '{print $7}')
n_draws=$(echo "$MODEL_ROW" | awk -F',' '{print $8}')
n_total=$(echo "$MODEL_ROW" | awk -F',' '{print $9}')
discrete_win_rate=$(echo "$MODEL_ROW" | awk -F',' '{print $10}')
length_controlled_winrate=$(echo "$MODEL_ROW" | awk -F',' '{print $11}')
lc_standard_error=$(echo "$MODEL_ROW" | awk -F',' '{print $12}')

# Handle empty values
win_rate=${win_rate:-0.0}
standard_error=${standard_error:-0.0}
avg_length=${avg_length:-0}
n_wins=${n_wins:-0}
n_wins_base=${n_wins_base:-0}
n_draws=${n_draws:-0}
n_total=${n_total:-0}
discrete_win_rate=${discrete_win_rate:-0.0}
length_controlled_winrate=${length_controlled_winrate:-0.0}
lc_standard_error=${lc_standard_error:-0.0}

# Create JSON output with all available metrics
cat > "$OUTPUT_FILE" << EOF
{
  "results": {
    "win_rate": $win_rate,
    "standard_error": $standard_error,
    "avg_length": $avg_length,
    "n_wins": $n_wins,
    "n_wins_base": $n_wins_base,
    "n_draws": $n_draws,
    "n_total": $n_total,
    "discrete_win_rate": $discrete_win_rate,
    "length_controlled_winrate": $length_controlled_winrate,
    "lc_standard_error": $lc_standard_error
  }
}
EOF

echo "Results saved to $OUTPUT_FILE:"

set +x
