#!/bin/bash

# script to summarize the most commonly-used results in a standard way.

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <DIR>"
    exit 1
fi
DIR=$1

echo -n "arc_challenge: " ; cat $DIR/arc_challenge.json | jq '.results.arc_challenge.acc_norm' | awk '{print $1 * 100}'
echo -n "    hellaswag: " ; cat $DIR/hellaswag.json | jq .results.hellaswag.acc_norm | awk '{print $1 * 100}'
echo -n "         mmlu: " ; cat $DIR/mmlu.json | jq '.results | .[] | .acc' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n * 100; }' 
echo -n "truthfulqa_mc: " ; cat $DIR/truthfulqa_mc.json | jq .results.truthfulqa_mc.mc2 | awk '{print $1 * 100}'
echo -n "   winogrande: " ; cat $DIR/winogrande.json | jq .results.winogrande.acc | awk '{print $1 * 100}'
echo -n "        gsm8k: " ; cat $DIR/gsm8k.json | jq .results.gsm8k.acc | awk '{print $1 * 100}'

echo
echo finbench:
cat $DIR/finbench.json | jq -r '.results | to_entries | .[] | [.key, .value.multiple_choice_grade] | @csv'


