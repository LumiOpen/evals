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

echo

if [ -f $DIR/arc_easy.json ] ; then
    echo -n "             arc_easy: "
    cat $DIR/arc_easy.json | jq .results.arc_easy.acc | awk '{print $1 * 100}'
fi

if [ -f $DIR/boolq.json ] ; then
    echo -n "                boolq: "
    cat $DIR/boolq.json | jq .results.boolq.acc | awk '{print $1 * 100}'
fi

if [ -f $DIR/nq_open.json ] ; then
    echo -n "              nq_open: "
    cat $DIR/nq_open.json | jq .results.nq_open.em | awk '{print $1 * 100}'
fi

if [ -f $DIR/openbookqa.json ] ; then
    echo -n "     openbookqa (acc): "
    cat $DIR/openbookqa.json | jq .results.openbookqa.acc | awk '{print $1 * 100}'
fi


if [ -f $DIR/piqa.json ] ; then
    echo -n "           piqa (acc): "
    cat $DIR/piqa.json | jq .results.piqa.acc | awk '{print $1 * 100}'
fi

if [ -f $DIR/toxigen.json ] ; then
    echo -n "        toxigen (acc): "
    cat $DIR/toxigen.json | jq .results.toxigen.acc | awk '{print $1 * 100}'
fi

if [ -f $DIR/triviaqa.json ] ; then
    echo -n "             triviaqa: "
    cat $DIR/triviaqa.json | jq .results.triviaqa.em | awk '{print $1 * 100}'
fi

# squad2 reports many different results, we need to determine which ones to use
if [ -f $DIR/squad2.json ] ; then
    echo
    echo -n "       squad2 (exact): "
    cat $DIR/squad2.json | jq .results.squad2.exact
    echo -n "          squad2 (f1): "
    cat $DIR/squad2.json | jq .results.squad2.f1
    echo -n "squad2 (HasAns_exact): "
    cat $DIR/squad2.json | jq .results.squad2.HasAns_exact
    echo -n "   squad2 (HasAns_f1): "
    cat $DIR/squad2.json | jq .results.squad2.HasAns_f1
    echo -n " squad2 (NoAns_exact): "
    cat $DIR/squad2.json | jq .results.squad2.NoAns_exact
    echo -n "    squad2 (NoAns_f1): "
    cat $DIR/squad2.json | jq .results.squad2.NoAns_f1
    echo -n "  squad2 (best_exact): "
    cat $DIR/squad2.json | jq .results.squad2.best_exact
    echo -n "     squad2 (best_f1): "
    cat $DIR/squad2.json | jq .results.squad2.best_f1
fi


### add math and drop

echo

if [ -f $DIR/humaneval_pass@1.json ] ; then
    echo -n "     humaneval_pass@1: "
    cat $DIR/humaneval_pass@1.json | jq '.humaneval."pass@1"' | awk '{print $1 * 100 }'
fi

if [ -f $DIR/humaneval_pass@10.json ] ; then
    echo -n "    humaneval_pass@10: "
    cat $DIR/humaneval_pass@10.json | jq '.humaneval."pass@10"' | awk '{print $1 * 100 }'
fi
if [ -f $DIR/mbpp_pass@1.json ] ; then
    echo -n "          mbpp_pass@1: "
    cat $DIR/mbpp_pass@1.json | jq '.mbpp."pass@1"' | awk '{print $1 * 100 }'
fi

if [ -f $DIR/mbpp_pass@10.json ] ; then
    echo -n "         mbpp_pass@10: "
    cat $DIR/mbpp_pass@10.json | jq '.mbpp."pass@10"' | awk '{print $1 * 100 }'
fi

echo 

if [ -f $DIR/finbench_3shot.json ] ; then 
    echo finbench_3shot:
    cat $DIR/finbench_3shot.json | jq -r '.results | to_entries | .[] | [.key, .value.multiple_choice_grade] | @csv'
fi


