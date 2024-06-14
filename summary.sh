#!/bin/bash

# script to summarize the most commonly-used results in a standard way.

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <DIR>"
    exit 1
fi
DIR=$1

echo -n "    arc_challenge: "
if [ -f $DIR/arc_challenge.json ] ; then 
    cat $DIR/arc_challenge.json | jq '.results.arc_challenge.acc_norm // .results.arc_challenge["acc_norm,none"]' | awk '{print $1 * 100}'
else
    echo na
fi
if [ -f $DIR/arc_challenge2.json ] ; then 
    echo -n "   arc_challenge2: "
    cat $DIR/arc_challenge2.json | jq '.results.arc_challenge.acc_norm // .results.arc_challenge["acc_norm,none"]' | awk '{print $1 * 100}'
fi

echo -n "        hellaswag: "
if [ -f $DIR/hellaswag.json ] ; then
    cat $DIR/hellaswag.json | jq '.results.hellaswag.acc_norm // .results.hellaswag["acc_norm,none"]' | awk '{print $1 * 100}'
else
    echo na
fi
if [ -f $DIR/hellaswag2.json ] ; then
    echo -n "       hellaswag2: "
    cat $DIR/hellaswag2.json | jq '.results.hellaswag.acc_norm // .results.hellaswag["acc_norm,none"]' | awk '{print $1 * 100}'
fi

echo -n "             mmlu: "
if [ -f $DIR/mmlu.json ] ; then
    cat $DIR/mmlu.json | jq '[if .results.mmlu | has("acc,none") then .results.mmlu["acc,none"] else .results | .[] | .acc end] | add / length * 100'
else
    echo na
fi
if [ -f $DIR/mmlu2.json ] ; then
    echo -n "            mmlu2: "
    cat $DIR/mmlu2.json | jq '.results.mmlu."acc,none"' | awk '{print $1 * 100}'
fi

echo -n "    truthfulqa_mc: "
if [ -f $DIR/truthfulqa_mc.json ] ; then 
    cat $DIR/truthfulqa_mc.json | jq '.results.truthfulqa_mc.mc2 // .results.truthfulqa_mc2["acc,none"]' | awk '{print $1 * 100}'
else
    echo na
fi
if [ -f $DIR/truthfulqa_mc2.json ] ; then 
    echo -n "   truthfulqa_mc2: "
    cat $DIR/truthfulqa_mc.json | jq '.results.truthfulqa_mc.mc2 // .results.truthfulqa_mc2["acc,none"]' | awk '{print $1 * 100}'
fi

echo -n "       winogrande: "
if [ -f $DIR/winogrande.json ] ; then
   cat $DIR/winogrande.json | jq '.results.winogrande.acc // .results.winogrande["acc,none"]' | awk '{print $1 * 100}'
else
    echo na
fi
if [ -f $DIR/winogrande2.json ] ; then
    echo -n "      winogrande2: "
   cat $DIR/winogrande2.json | jq '.results.winogrande.acc // .results.winogrande["acc,none"]' | awk '{print $1 * 100}'
fi

echo -n "            gsm8k: "
if [ -f $DIR/gsm8k.json ] ; then
    cat $DIR/gsm8k.json | jq '.results.gsm8k.acc // .results.gsm8k["exact_match,flexible-extract"]' | awk '{print $1 * 100}'
else
    echo na
fi
if [ -f $DIR/gsm8k2.json ] ; then
    echo -n "           gsm8k2: "
    cat $DIR/gsm8k2.json | jq '.results.gsm8k.acc // .results.gsm8k["exact_match,flexible-extract"]' | awk '{print $1 * 100}'
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
elif [ -f $DIR/arc_challenge_mt_fi.json ] ; then 
    cat $DIR/arc_challenge_mt_fi.json | jq '.results.arc_challenge_mt_fi["acc_norm,none"]' | awk '{print $1 * 100}'
else
    echo na
fi


echo

echo -n " arc_challenge_da: "
if [ -f $DIR/arc_challenge_da.json ] ; then 
    cat $DIR/arc_challenge_da.json | jq '.results.arc_challenge_da.acc_norm' | awk '{print $1 * 100}'
elif [ -f $DIR/arc_challenge_mt_da.json ] ; then 
    cat $DIR/arc_challenge_mt_da.json | jq '.results.arc_challenge_mt_da["acc_norm,none"]' | awk '{print $1 * 100}'
else
    echo na
fi
echo -n " arc_challenge_is: "
if [ -f $DIR/arc_challenge_is.json ] ; then 
    cat $DIR/arc_challenge_is.json | jq '.results.arc_challenge_is.acc_norm' | awk '{print $1 * 100}'
elif [ -f $DIR/arc_challenge_mt_is.json ] ; then 
    cat $DIR/arc_challenge_mt_is.json | jq '.results.arc_challenge_mt_is["acc_norm,none"]' | awk '{print $1 * 100}'
else
    echo na
fi
echo -n " arc_challenge_nb: "
if [ -f $DIR/arc_challenge_nb.json ] ; then 
    cat $DIR/arc_challenge_nb.json | jq '.results.arc_challenge_nb.acc_norm' | awk '{print $1 * 100}'
elif [ -f $DIR/arc_challenge_mt_nb.json ] ; then 
    cat $DIR/arc_challenge_mt_nb.json | jq '.results.arc_challenge_mt_nb["acc_norm,none"]' | awk '{print $1 * 100}'
else
    echo na
fi
echo -n " arc_challenge_sv: "
if [ -f $DIR/arc_challenge_sv.json ] ; then 
    cat $DIR/arc_challenge_sv.json | jq '.results.arc_challenge_sv.acc_norm' | awk '{print $1 * 100}'
elif [ -f $DIR/arc_challenge_mt_sv.json ] ; then 
    cat $DIR/arc_challenge_mt_sv.json | jq '.results.arc_challenge_mt_sv["acc_norm,none"]' | awk '{print $1 * 100}'
else
    echo na
fi

echo

echo -n " arc_challenge_de: "
if [ -f $DIR/arc_challenge_de.json ] ; then
    cat $DIR/arc_challenge_de.json | jq '.results.arc_challenge_de.acc_norm' | awk '{print $1 * 100}'
elif [ -f $DIR/arc_challenge_mt_de.json ] ; then 
    cat $DIR/arc_challenge_mt_de.json | jq '.results.arc_challenge_mt_de["acc_norm,none"]' | awk '{print $1 * 100}'
else
    echo na
fi
echo -n " arc_challenge_el: "
if [ -f $DIR/arc_challenge_el.json ] ; then
    cat $DIR/arc_challenge_el.json | jq '.results.arc_challenge_el.acc_norm' | awk '{print $1 * 100}'
elif [ -f $DIR/arc_challenge_mt_el.json ] ; then 
    cat $DIR/arc_challenge_mt_el.json | jq '.results.arc_challenge_mt_el["acc_norm,none"]' | awk '{print $1 * 100}'
else
    echo na
fi
echo -n " arc_challenge_es: "
if [ -f $DIR/arc_challenge_es.json ] ; then
    cat $DIR/arc_challenge_es.json | jq '.results.arc_challenge_es.acc_norm' | awk '{print $1 * 100}'
elif [ -f $DIR/arc_challenge_mt_es.json ] ; then 
    cat $DIR/arc_challenge_mt_es.json | jq '.results.arc_challenge_mt_es["acc_norm,none"]' | awk '{print $1 * 100}'
else
    echo na
fi
echo -n " arc_challenge_hu: "
if [ -f $DIR/arc_challenge_hu.json ] ; then
    cat $DIR/arc_challenge_hu.json | jq '.results.arc_challenge_hu.acc_norm' | awk '{print $1 * 100}'
elif [ -f $DIR/arc_challenge_mt_hu.json ] ; then 
    cat $DIR/arc_challenge_mt_hu.json | jq '.results.arc_challenge_mt_hu["acc_norm,none"]' | awk '{print $1 * 100}'
else
    echo na
fi
echo -n " arc_challenge_it: "
if [ -f $DIR/arc_challenge_it.json ] ; then
    cat $DIR/arc_challenge_it.json | jq '.results.arc_challenge_it.acc_norm' | awk '{print $1 * 100}'
elif [ -f $DIR/arc_challenge_mt_it.json ] ; then 
    cat $DIR/arc_challenge_mt_it.json | jq '.results.arc_challenge_mt_it["acc_norm,none"]' | awk '{print $1 * 100}'
else
    echo na
fi
echo -n " arc_challenge_pl: "
if [ -f $DIR/arc_challenge_pl.json ] ; then
    cat $DIR/arc_challenge_pl.json | jq '.results.arc_challenge_pl.acc_norm' | awk '{print $1 * 100}'
elif [ -f $DIR/arc_challenge_mt_pl.json ] ; then 
    cat $DIR/arc_challenge_mt_pl.json | jq '.results.arc_challenge_mt_pl["acc_norm,none"]' | awk '{print $1 * 100}'
else
    echo na
fi
echo -n " arc_challenge_pt: "
if [ -f $DIR/arc_challenge_pt.json ] ; then
    cat $DIR/arc_challenge_pt.json | jq '.results.arc_challenge_pt.acc_norm' | awk '{print $1 * 100}'
elif [ -f $DIR/arc_challenge_mt_pt.json ] ; then 
    cat $DIR/arc_challenge_mt_pt.json | jq '.results.arc_challenge_mt_pt["acc_norm,none"]' | awk '{print $1 * 100}'
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

#   cat $DIR/winogrande2.json | jq '.results.winogrande.acc // .results.winogrande["acc,none"]' | awk '{print $1 * 100}'

echo -n "         arc_easy: "
if [ -f $DIR/arc_easy.json ] ; then
    cat $DIR/arc_easy.json | jq '.results.arc_easy.acc // .results.arc_easy["acc,none"]' | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "            boolq: "
if [ -f $DIR/boolq.json ] ; then
    cat $DIR/boolq.json | jq '.results.boolq.acc // .results.boolq["acc,none"]' | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "          nq_open: "
if [ -f $DIR/nq_open.json ] ; then
    cat $DIR/nq_open.json | jq '.results.nq_open.em // .results.nq_open["exact_match,remove_whitespace"]' | awk '{print $1 * 100}'
else
    echo na
fi

echo -n " openbookqa (acc): "
if [ -f $DIR/openbookqa.json ] ; then
    cat $DIR/openbookqa.json | jq '.results.openbookqa.acc // .results.openbookqa["acc,none"]' | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "       piqa (acc): "
if [ -f $DIR/piqa.json ] ; then
    cat $DIR/piqa.json | jq '.results.piqa.acc // .results.piqa["acc,none"]' | awk '{print $1 * 100}'
else
    echo na
fi

echo -n "      squad2 (f1): "
if [ -f $DIR/squad2.json ] ; then
    cat $DIR/squad2.json | jq '.results.squad2.f1 // .results.squadv2["f1,none"]'
else
    echo na
fi

echo -n "         triviaqa: "
if [ -f $DIR/triviaqa.json ] ; then
    cat $DIR/triviaqa.json | jq '.results.triviaqa.em // .results.triviaqa["exact_match,remove_whitespace"]' | awk '{print $1 * 100}'
else
    echo na
fi

echo

echo -n "    toxigen (acc): "
if [ -f $DIR/toxigen.json ] ; then
    cat $DIR/toxigen.json | jq '.results.toxigen.acc // .results.toxigen["acc,none"]' | awk '{print $1 * 100}'
else
    echo na
fi


