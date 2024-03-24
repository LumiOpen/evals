#!/bin/bash

# script to summarize the most commonly-used results in a standard way.

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <DIR>"
    exit 1
fi
DIR=$1

echo -n "    arc_challenge: "
if [ -f $DIR/arc_challenge.json ] ; then 
    cat $DIR/arc_challenge.json | jq '.results.arc_challenge.acc_norm' | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "        hellaswag: "
if [ -f $DIR/hellaswag.json ] ; then
    cat $DIR/hellaswag.json | jq .results.hellaswag.acc_norm | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "             mmlu: "
if [ -f $DIR/mmlu.json ] ; then
    cat $DIR/mmlu.json | jq '.results | .[] | .acc' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n * 100; }' 
else
    echo na
fi

echo -n "    truthfulqa_mc: "
if [ -f $DIR/truthfulqa_mc.json ] ; then 
    cat $DIR/truthfulqa_mc.json | jq .results.truthfulqa_mc.mc2 | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "       winogrande: "
if [ -f $DIR/winogrande.json ] ; then
   cat $DIR/winogrande.json | jq .results.winogrande.acc | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "            gsm8k: "
if [ -f $DIR/gsm8k.json ] ; then
    cat $DIR/gsm8k.json | jq .results.gsm8k.acc | awk '{print $1 * 100}'
else
    echo na
fi

echo

echo -n "   finbench_3shot: "
if [ -f $DIR/finbench_3shot.json ] ; then 
    #cat $DIR/finbench_3shot.json | jq -r '.results | to_entries | .[] | [.key, .value.multiple_choice_grade] | @csv'
    cat $DIR/finbench_3shot.json | jq  '.results | 
  [
    .bigbench_1_digit_addition.multiple_choice_grade,
    .bigbench_1_digit_division.multiple_choice_grade,
    .bigbench_1_digit_multiplication.multiple_choice_grade,
    .bigbench_1_digit_subtraction.multiple_choice_grade,
    .bigbench_2_digit_addition.multiple_choice_grade,
    .bigbench_2_digit_division.multiple_choice_grade,
    .bigbench_2_digit_multiplication.multiple_choice_grade,
    .bigbench_2_digit_subtraction.multiple_choice_grade,
    .bigbench_3_digit_addition.multiple_choice_grade,
    .bigbench_3_digit_division.multiple_choice_grade,
    .bigbench_3_digit_multiplication.multiple_choice_grade,
    .bigbench_3_digit_subtraction.multiple_choice_grade,
    .bigbench_4_digit_addition.multiple_choice_grade,
    .bigbench_4_digit_division.multiple_choice_grade,
    .bigbench_4_digit_multiplication.multiple_choice_grade,
    .bigbench_4_digit_subtraction.multiple_choice_grade,
    .bigbench_5_digit_addition.multiple_choice_grade,
    .bigbench_5_digit_division.multiple_choice_grade,
    .bigbench_5_digit_multiplication.multiple_choice_grade,
    .bigbench_5_digit_subtraction.multiple_choice_grade
  ] as $math_ops | {
  other: [
    .bigbench_analogies.multiple_choice_grade,
    .bigbench_emotions.multiple_choice_grade,
    .bigbench_empirical_judgments.multiple_choice_grade,
    .bigbench_general_knowledge.multiple_choice_grade,
    .bigbench_harmless.multiple_choice_grade,
    .bigbench_helpful.multiple_choice_grade,
    .bigbench_honest.multiple_choice_grade,
    .bigbench_intent_recognition.multiple_choice_grade,
    .bigbench_misconceptions.multiple_choice_grade,
    .bigbench_one_sentence.multiple_choice_grade,
    .bigbench_one_sentence_no_prompt.multiple_choice_grade,
    .bigbench_other.multiple_choice_grade,
    .bigbench_paraphrase.multiple_choice_grade,
    .bigbench_sentence_ambiguity.multiple_choice_grade,
    .bigbench_similarities_abstraction.multiple_choice_grade,
    .bigbench_two_sentences.multiple_choice_grade,
    (($math_ops) | add / length)
  ]
} | {
  overall_average: ((.other) | add / length)
} | .overall_average' | awk '{print $1 * 100}'
else
    echo na
fi
echo -n " arc_challenge_fi: "
if [ -f $DIR/arc_challenge_fi.json ] ; then 
    cat $DIR/arc_challenge_fi.json | jq '.results.arc_challenge_fi.acc_norm' | awk '{print $1 * 100}'
else
    echo na
fi


echo

echo -n " humaneval_pass@1: "
if [ -f $DIR/humaneval_pass@1.json ] ; then
    cat $DIR/humaneval_pass@1.json | jq '.humaneval."pass@1"' | awk '{print $1 * 100 }'
else
    echo na
fi

echo -n "humaneval_pass@10: "
if [ -f $DIR/humaneval_pass@10.json ] ; then
    cat $DIR/humaneval_pass@10.json | jq '.humaneval."pass@10"' | awk '{print $1 * 100 }'
else
    echo na
fi
echo -n "      mbpp_pass@1: "
if [ -f $DIR/mbpp_pass@1.json ] ; then
    cat $DIR/mbpp_pass@1.json | jq '.mbpp."pass@1"' | awk '{print $1 * 100 }'
else
    echo na
fi

echo -n "     mbpp_pass@10: "
if [ -f $DIR/mbpp_pass@10.json ] ; then
    cat $DIR/mbpp_pass@10.json | jq '.mbpp."pass@10"' | awk '{print $1 * 100 }'
else
    echo na
fi

echo

echo -n "         arc_easy: "
if [ -f $DIR/arc_easy.json ] ; then
    cat $DIR/arc_easy.json | jq .results.arc_easy.acc | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "            boolq: "
if [ -f $DIR/boolq.json ] ; then
    cat $DIR/boolq.json | jq .results.boolq.acc | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "          nq_open: "
if [ -f $DIR/nq_open.json ] ; then
    cat $DIR/nq_open.json | jq .results.nq_open.em | awk '{print $1 * 100}'
else
    echo na
fi

echo -n " openbookqa (acc): "
if [ -f $DIR/openbookqa.json ] ; then
    cat $DIR/openbookqa.json | jq .results.openbookqa.acc | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "       piqa (acc): "
if [ -f $DIR/piqa.json ] ; then
    cat $DIR/piqa.json | jq .results.piqa.acc | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "      squad2 (f1): "
if [ -f $DIR/squad2.json ] ; then
    cat $DIR/squad2.json | jq .results.squad2.f1
else
    echo na
fi

echo -n "         triviaqa: "
if [ -f $DIR/triviaqa.json ] ; then
    cat $DIR/triviaqa.json | jq .results.triviaqa.em | awk '{print $1 * 100}'
else
    echo na
fi

echo

echo -n "    toxigen (acc): "
if [ -f $DIR/toxigen.json ] ; then
    cat $DIR/toxigen.json | jq .results.toxigen.acc | awk '{print $1 * 100}'
else
    echo na
fi


