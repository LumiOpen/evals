#!/bin/bash
# One-time setup for HELMET + LongPPL evaluation on TensorWave.
# Run on a node with access to /shared_silo/scratch:
#   ssh tus1-vm-amd-misc-05
#   bash setup.sh
set -euo pipefail

SCRATCH="${SCRATCH:-/shared_silo/scratch/$USER}"

echo "=== TensorWave eval setup ==="
echo "Scratch dir: $SCRATCH"

# 1. Clone LumiOpen lm-eval fork (crosslingual-longppl branch)
LMEVAL_DIR=$SCRATCH/lm-eval-longppl
if [ -d "$LMEVAL_DIR" ]; then
    echo "lm-eval-longppl already at $LMEVAL_DIR, pulling latest..."
    cd "$LMEVAL_DIR" && git pull && cd -
else
    echo "Cloning LumiOpen lm-eval-harness (crosslingual-longppl branch)..."
    git clone -b feature/crosslingual-longppl \
        https://github.com/LumiOpen/lm-evaluation-harness.git "$LMEVAL_DIR"
fi

# 2. Clone HELMET
HELMET_DIR=$SCRATCH/HELMET
if [ -d "$HELMET_DIR" ]; then
    echo "HELMET already at $HELMET_DIR, pulling latest..."
    cd "$HELMET_DIR" && git pull && cd -
else
    echo "Cloning HELMET..."
    git clone https://github.com/princeton-nlp/HELMET.git "$HELMET_DIR"
fi

# 3. Download HELMET data (~34 GB)
if [ -d "$HELMET_DIR/data" ] && [ "$(ls -A "$HELMET_DIR/data" 2>/dev/null)" ]; then
    echo "HELMET data already exists, skipping download."
else
    echo "Downloading HELMET data (~34 GB)..."
    cd "$HELMET_DIR"
    wget -c https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/main/data.tar.gz
    echo "Extracting data..."
    tar -xzf data.tar.gz
    rm -f data.tar.gz
    cd -
fi

# 4. Create output directories
mkdir -p "$SCRATCH/eval_results/longppl"
mkdir -p "$SCRATCH/eval_results/helmet"
mkdir -p "$HELMET_DIR/output"

echo ""
echo "=== Setup complete ==="
echo "HELMET repo:    $HELMET_DIR"
echo "lm-eval fork:   $LMEVAL_DIR"
echo ""
echo "Submit evals with:"
echo "  sbatch --export=ALL,MODEL=<model_path_or_hf_id> eval_helmet.sbatch"
echo "  sbatch --export=ALL,MODEL=<model_path_or_hf_id> eval_longppl.sbatch"
