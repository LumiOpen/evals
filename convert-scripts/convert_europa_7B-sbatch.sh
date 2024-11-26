#!/bin/bash

#SBATCH --job-name=conv
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --partition=dev-g
#SBATCH --time=00-00:30:00
#SBATCH --gpus-per-node=mi250:1
#SBATCH --account=project_462000353
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

SCRIPT_DIR="/scratch/project_462000353/risto/Megatron-LM-319"
CONTAINER="/scratch/project_462000353/containers/flashattention_v2_new"
SING_BIND="/scratch/project_462000353"
TOKENIZER="/scratch/project_462000353/europa-tokenizer"

CONFIG_FILE="/scratch/project_462000353/risto/Megatron-LM-319/conversion_scripts/configs/europa_7B_config.json"
if [ -z "$1" ]; 
    then
    echo "You need to pass the checkpoint path as an argument!"
fi
set -euo pipefail

# Give checkpoint path and output dir as positional args
CHECKPOINT_PATH=$1
OUTPUT_ROOT="/scratch/project_462000353/risto/europa/converted_models"
OUTPUT_DIR=$OUTPUT_ROOT/europa_7B_$(basename ${CHECKPOINT_PATH})_bfloat16



CMD="python $SCRIPT_DIR/tools/checkpoint/custom/convert_to_llama.py \
    --path_to_unmerged_checkpoint $CHECKPOINT_PATH \
    --config_file $CONFIG_FILE \
    --tokenizer $TOKENIZER \
    --output_dir $OUTPUT_DIR
    "

srun \
    singularity exec \
    -B "$SING_BIND" \
                -B "$PWD" \
                "$CONTAINER" \
                bash -c "source /opt/miniconda3/bin/activate pytorch; echo $PYTHONUSERBASE; python -V ;$CMD"
