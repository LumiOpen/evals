#!/bin/bash
# LongPPL Run Evaluation Stage
# Runs the LongPPL evaluation using the model and parameters set up in the setup stage

set -euo pipefail

echo "=== LongPPL Run Evaluation Stage ==="

# Source variables from setup stage
cd /workspace

# Use meta-llama/Llama-3.1-8B as evaluator (same as paper)
# This model identifies key tokens; then we compute LongPPL for target model on those tokens
EVALUATOR_MODEL="meta-llama/Llama-3.1-8B"

echo "=== Starting LongPPL evaluation ==="
echo "Target model: $MODEL_LOCAL"
echo "Evaluator model: $EVALUATOR_MODEL"
echo "Context length: $CONTEXT_LENGTH"
echo "Dataset samples: $DATASET_SAMPLES"
echo "Alpha threshold: $ALPHA"
echo "Beta threshold: $BETA"
echo "Output directory: $OUTPUT_DIR"
echo "Note: Using default max-length=32768 to match paper evaluation setup"
echo

python longppl/perplexity/perplexity.py \
  --dataset emozilla/govreport-test-tokenized \
  --tokenized \
  --dataset-min-tokens "$CONTEXT_LENGTH" \
  --samples "$DATASET_SAMPLES" \
  --model "$MODEL_LOCAL" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --mode online \
  --trunc-len 4096 \
  --sliding-window 1024 \
  --alpha "$ALPHA" \
  --beta "$BETA" \
  2>&1 | tee "$OUTPUT_DIR/longppl_${CONTEXT_LENGTH}.log"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
  echo "✓ LongPPL evaluation completed successfully"
else
  echo "✗ LongPPL evaluation failed with exit code $EXIT_CODE"
  exit $EXIT_CODE
fi
