#!/bin/bash
QUEUE=standard-g
PROJECT=project_462000963
MODEL=$1
TASK=$2
SEQ_LENGTH=$3

echo "Running task $TASK with seq length $SEQ_LENGTH on model $MODEL"
python3 main.py --time 12:00:00 --project $PROJECT --partition $QUEUE --ruler_seq_length $SEQ_LENGTH --model $MODEL $TASK