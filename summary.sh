#!/bin/bash

# script to summarize the most commonly-used results in a standard way.

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <DIR>"
    exit 1
fi
DIR=$1

echo -n "arc_challenge: "
if [ -f $DIR/arc_challenge.json ] ; then 
    cat $DIR/arc_challenge.json | jq '.results.arc_challenge.acc_norm' | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "    hellaswag: "
if [ -f $DIR/hellaswag.json ] ; then
    cat $DIR/hellaswag.json | jq .results.hellaswag.acc_norm | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "         mmlu: "
if [ -f $DIR/mmlu.json ] ; then
    cat $DIR/mmlu.json | jq '.results | .[] | .acc' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n * 100; }' 
else
    echo na
fi

echo -n "truthfulqa_mc: "
if [ -f $DIR/truthfulqa_mc.json ] ; then 
    cat $DIR/truthfulqa_mc.json | jq .results.truthfulqa_mc.mc2 | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "   winogrande: "
if [ -f $DIR/winogrande.json ] ; then
   cat $DIR/winogrande.json | jq .results.winogrande.acc | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "        gsm8k: "
if [ -f $DIR/gsm8k.json ] ; then
    cat $DIR/gsm8k.json | jq .results.gsm8k.acc | awk '{print $1 * 100}'
else
    echo na
fi

if [ -f $DIR/finbench.json ] ; then 
    echo finbench:
    cat $DIR/finbench.json | jq -r '.results | to_entries | .[] | [.key, .value.multiple_choice_grade] | @csv'
fi


