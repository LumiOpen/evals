#!/bin/bash

# script to summarize the most commonly-used results in a standard way.

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <DIR>"
    exit 1
fi
DIR=$1

pick_file() {
    local stem="$1"

    local exact="$DIR/${stem}.json"
    local exact_vllm="$DIR/vllm_${stem}.json"

    # Prefer legacy files when they exist and are non-empty.
    if [ -s "$exact" ]; then
        echo "$exact"
        return
    fi
    if [ -s "$exact_vllm" ]; then
        echo "$exact_vllm"
        return
    fi

    # Otherwise accept timestamp-suffixed files and pick newest non-empty.
    # Important: avoid accidentally matching *other* task stems that merely share a prefix.
    # Example: for stem="arc_challenge", we want vllm_arc_challenge_<ts>.json
    # but NOT vllm_arc_challenge_mt_fi_<ts>.json.
    local candidates=()
    local nonempty=()
    local filtered=()

    shopt -s nullglob
    candidates=(
        "$DIR/${stem}"_*.json
        "$DIR/${stem}"-*.json
        "$DIR/vllm_${stem}"_*.json
        "$DIR/vllm_${stem}"-*.json
    )
    shopt -u nullglob

    for f in "${candidates[@]}"; do
        [ -s "$f" ] && nonempty+=("$f")
    done

    # Prefer files where the suffix right after "${stem}_" or "${stem}-" starts with a digit
    # (our timestamped outputs begin with YYYY-...). This filters out e.g. "${stem}_mt_fi_...".
    for f in "${nonempty[@]}"; do
        local base base_no_prefix rest
        base="$(basename -- "$f")"
        base_no_prefix="$base"
        base_no_prefix="${base_no_prefix#vllm_}"

        if [[ "$base_no_prefix" == "${stem}_"* ]]; then
            rest="${base_no_prefix#${stem}_}"
            [[ "$rest" =~ ^[0-9] ]] && filtered+=("$f")
        elif [[ "$base_no_prefix" == "${stem}-"* ]]; then
            rest="${base_no_prefix#${stem}-}"
            [[ "$rest" =~ ^[0-9] ]] && filtered+=("$f")
        fi
    done

    if [ ${#filtered[@]} -gt 0 ]; then
        nonempty=("${filtered[@]}")
    fi

    if [ ${#nonempty[@]} -gt 0 ]; then
        ls -1t "${nonempty[@]}" 2>/dev/null | head -n 1
        return
    fi

    echo ""
}

# Fallback: pick newest non-empty JSON in $DIR where jq_filter yields a value
pick_file_by_jq() {
    local jq_filter="$1"

    local candidates=()
    local nonempty=()
    local matched=()

    shopt -s nullglob
    candidates=("$DIR"/*.json)
    shopt -u nullglob

    for f in "${candidates[@]}"; do
        [ -s "$f" ] && nonempty+=("$f")
    done

    for f in "${nonempty[@]}"; do
        local v
        v="$(jq -r "$jq_filter" "$f" 2>/dev/null)"
        if [ -n "$v" ] && [ "$v" != "null" ]; then
            matched+=("$f")
        fi
    done

    if [ ${#matched[@]} -gt 0 ]; then
        ls -1t "${matched[@]}" 2>/dev/null | head -n 1
        return
    fi

    echo ""
}

print_metric() {
    # Args: label, stem, jq_filter, scale
    local label="$1"
    local stem="$2"
    local jq_filter="$3"
    local scale="${4:-1}"

    echo -n "$label"
    local file
    file="$(pick_file "$stem")"

    local val
    if [ -n "$file" ]; then
        val="$(jq -r "$jq_filter" "$file" 2>/dev/null)"
    fi

    # If stem selection fails (missing file or metric not in that file),
    # fall back to scanning for a JSON where jq_filter yields a value.
    if [ -z "$file" ] || [ -z "$val" ] || [ "$val" = "null" ]; then
        file="$(pick_file_by_jq "$jq_filter")"
        if [ -z "$file" ]; then
            echo "na"
            return
        fi
        val="$(jq -r "$jq_filter" "$file" 2>/dev/null)"
        if [ -z "$val" ] || [ "$val" = "null" ]; then
            echo "na"
            return
        fi
    fi

    awk -v v="$val" -v s="$scale" 'BEGIN { print v * s }'
}

print_metric_if_exists() {
    # Like print_metric, but prints nothing if no file exists.
    # Args: label, stem, jq_filter, scale
    local label="$1"
    local stem="$2"
    local jq_filter="$3"
    local scale="${4:-1}"

    local file
    file="$(pick_file "$stem")"
    [ -z "$file" ] && return 0

    print_metric "$label" "$stem" "$jq_filter" "$scale"
}

# --- core English benchmarks ---
print_metric "    arc_challenge: " "arc_challenge" '.results.arc_challenge.acc_norm // .results.arc_challenge["acc_norm,none"]' 100
print_metric_if_exists "   arc_challenge2: " "arc_challenge2" '.results.arc_challenge.acc_norm // .results.arc_challenge["acc_norm,none"]' 100

print_metric "        hellaswag: " "hellaswag" '.results.hellaswag.acc_norm // .results.hellaswag["acc_norm,none"]' 100
print_metric_if_exists "       hellaswag2: " "hellaswag2" '.results.hellaswag.acc_norm // .results.hellaswag["acc_norm,none"]' 100

print_metric "       goldenswag: " "goldenswag" '.results.goldenswag.acc_norm // .results.goldenswag["acc_norm,none"]' 100

# mmlu is special: historically either .results.mmlu["acc,none"] or per-subtask .results[].acc
echo -n "             mmlu: "
mmlu_file="$(pick_file "mmlu")"
if [ -n "$mmlu_file" ]; then
    jq -r '[if (.results.mmlu? | has("acc,none")) then .results.mmlu["acc,none"] else (.results | .[] | .acc) end] | add / length * 100' "$mmlu_file" 2>/dev/null \
        | awk '{print $1}'
else
    echo na
fi
print_metric_if_exists "            mmlu2: " "mmlu2" '.results.mmlu."acc,none"' 100

# truthfulqa_mc: keep original fallbacks but avoid cat|jq and allow timestamped/vllm names
print_metric "    truthfulqa_mc: " "truthfulqa_mc" '.results.truthfulqa_mc.mc2 // .results.truthfulqa_mc["acc,none"] // .results.truthfulqa_mc2["acc,none"]' 100
print_metric_if_exists "   truthfulqa_mc2: " "truthfulqa_mc2" '.results.truthfulqa_mc.mc2 // .results.truthfulqa_mc["acc,none"] // .results.truthfulqa_mc2["acc,none"]' 100

print_metric "       winogrande: " "winogrande" '.results.winogrande.acc // .results.winogrande["acc,none"]' 100
print_metric_if_exists "      winogrande2: " "winogrande2" '.results.winogrande.acc // .results.winogrande["acc,none"]' 100

print_metric "            gsm8k: " "gsm8k" '.results.gsm8k.acc // .results.gsm8k["exact_match,flexible-extract"]' 100
print_metric_if_exists "           gsm8k2: " "gsm8k2" '.results.gsm8k.acc // .results.gsm8k["exact_match,flexible-extract"]' 100

echo

# --- finbench (needs content sniff) ---
echo -n "     finbench_3shot: "
fin_file="$(pick_file "finbench_3shot")"
if [ -z "$fin_file" ]; then
    echo na
else
    if grep -q "FIN-bench" "$fin_file" >/dev/null 2>&1; then
        jq -r '.results | to_entries | map(.value."acc,none") | add / length' "$fin_file" 2>/dev/null | awk '{print $1 * 100}'
    else
        jq -r '.results |
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
        } | .overall_average' "$fin_file" 2>/dev/null | awk '{print $1 * 100}'
    fi
fi

# --- Finnish MT suite ---
print_metric "   arc_challenge_fi: " "arc_challenge_mt_fi" '.results.arc_challenge_fi.acc_norm // .results.arc_challenge_mt_fi["acc_norm,none"] // .results.arc_challenge_fi["acc_norm,none"]' 100
print_metric "    hellaswag_mt_fi: " "hellaswag_mt_fi" '.results.ogx_hellaswagx_fi."acc,none"' 100
print_metric "    goldenswag_mt_fi: " "goldenswag_mt_fi" '.results.ogx_goldenswagx_fi."acc,none"' 100
print_metric "         mmlu_mt_fi: " "mmlu_mt_fi" '.groups.ogx_mmlux_FI."acc,none"' 100
print_metric "truthfulqa_mc_mt_fi: " "truthfulqa_mc_mt_fi" '.results.ogx_truthfulqax_mc2_fi."acc,none"' 100
print_metric "        gsm8k_mt_fi: " "gsm8k_mt_fi" '.results.ogx_gsm8kx_fi."acc,none"' 100

echo

print_metric "    flores200_en_fi bleu: " "flores200_trans_en_fi" '.results."ogx_flores200-trans-eng_Latn-fin_Latn"."bleu_flores200,none"' 1
print_metric "    flores200_fi_en bleu: " "flores200_trans_fi_en" '.results."ogx_flores200-trans-fin_Latn-eng_Latn"."bleu_flores200,none"' 1
print_metric "    flores200_en_fi chrf: " "flores200_trans_en_fi" '.results."ogx_flores200-trans-eng_Latn-fin_Latn"."chrf,none"' 1
print_metric "    flores200_fi_en chrf: " "flores200_trans_fi_en" '.results."ogx_flores200-trans-fin_Latn-eng_Latn"."chrf,none"' 1

echo
# --- Danish MT suite ---
print_metric "   arc_challenge_da: " "arc_challenge_da" '.results.arc_challenge_da.acc_norm // .results.arc_challenge_mt_da["acc_norm,none"] // .results.arc_challenge_da["acc_norm,none"]' 100
print_metric "    hellaswag_mt_da: " "hellaswag_mt_da" '.results.ogx_hellaswagx_da."acc,none"' 100
print_metric "         mmlu_mt_da: " "mmlu_mt_da" '.groups.ogx_mmlux_DA."acc,none"' 100
print_metric "truthfulqa_mc_mt_da: " "truthfulqa_mc_mt_da" '.results.ogx_truthfulqax_mc2_da."acc,none"' 100
print_metric "        gsm8k_mt_da: " "gsm8k_mt_da" '.results.ogx_gsm8kx_da."acc,none"' 100
print_metric "    flores200_en_da: " "flores200_trans_en_da" '.results."ogx_flores200-trans-eng_Latn-dan_Latn"."bleu_flores200,none"' 1
print_metric "    flores200_da_en: " "flores200_trans_da_en" '.results."ogx_flores200-trans-dan_Latn-eng_Latn"."bleu_flores200,none"' 1

echo
# --- Icelandic ---
print_metric " arc_challenge_is: " "arc_challenge_is" '.results.arc_challenge_is.acc_norm // .results.arc_challenge_mt_is["acc_norm,none"] // .results.arc_challenge_is["acc_norm,none"]' 100
print_metric "    flores200_en_is: " "flores200_trans_en_is" '.results."ogx_flores200-trans-eng_Latn-isl_Latn"."bleu_flores200,none"' 1
print_metric "    flores200_is_en: " "flores200_trans_is_en" '.results."ogx_flores200-trans-isl_Latn-eng_Latn"."bleu_flores200,none"' 1

echo
# --- Norwegian Bokmal ---
print_metric " arc_challenge_nb: " "arc_challenge_nb" '.results.arc_challenge_nb.acc_norm // .results.arc_challenge_mt_nb["acc_norm,none"] // .results.arc_challenge_nb["acc_norm,none"]' 100
print_metric "    flores200_en_nb: " "flores200_trans_en_nb" '.results."ogx_flores200-trans-eng_Latn-nob_Latn"."bleu_flores200,none"' 1
print_metric "    flores200_nb_en: " "flores200_trans_nb_en" '.results."ogx_flores200-trans-nob_Latn-eng_Latn"."bleu_flores200,none"' 1

echo
# --- Swedish ---
print_metric "   arc_challenge_sv: " "arc_challenge_sv" '.results.arc_challenge_sv.acc_norm // .results.arc_challenge_mt_sv["acc_norm,none"] // .results.arc_challenge_sv["acc_norm,none"]' 100
print_metric "    hellaswag_mt_sv: " "hellaswag_mt_sv" '.results.ogx_hellaswagx_sv."acc,none"' 100
print_metric "         mmlu_mt_sv: " "mmlu_mt_sv" '.groups.ogx_mmlux_SV."acc,none"' 100
print_metric "truthfulqa_mc_mt_sv: " "truthfulqa_mc_mt_sv" '.results.ogx_truthfulqax_mc2_sv."acc,none"' 100
print_metric "        gsm8k_mt_sv: " "gsm8k_mt_sv" '.results.ogx_gsm8kx_sv."acc,none"' 100
print_metric "    flores200_en_sv: " "flores200_trans_en_sv" '.results."ogx_flores200-trans-eng_Latn-swe_Latn"."bleu_flores200,none"' 1
print_metric "    flores200_sv_en: " "flores200_trans_sv_en" '.results."ogx_flores200-trans-swe_Latn-eng_Latn"."bleu_flores200,none"' 1

echo
# --- arc_challenge (other languages) ---
print_metric " arc_challenge_de: " "arc_challenge_de" '.results.arc_challenge_de.acc_norm // .results.arc_challenge_mt_de["acc_norm,none"] // .results.arc_challenge_de["acc_norm,none"]' 100
print_metric " arc_challenge_el: " "arc_challenge_el" '.results.arc_challenge_el.acc_norm // .results.arc_challenge_mt_el["acc_norm,none"] // .results.arc_challenge_el["acc_norm,none"]' 100
print_metric " arc_challenge_es: " "arc_challenge_es" '.results.arc_challenge_es.acc_norm // .results.arc_challenge_mt_es["acc_norm,none"] // .results.arc_challenge_es["acc_norm,none"]' 100
print_metric " arc_challenge_fr: " "arc_challenge_fr" '.results.arc_challenge_fr.acc_norm // .results.arc_challenge_mt_fr["acc_norm,none"] // .results.arc_challenge_fr["acc_norm,none"]' 100
print_metric " arc_challenge_hu: " "arc_challenge_hu" '.results.arc_challenge_hu.acc_norm // .results.arc_challenge_mt_hu["acc_norm,none"] // .results.arc_challenge_hu["acc_norm,none"]' 100
print_metric " arc_challenge_it: " "arc_challenge_it" '.results.arc_challenge_it.acc_norm // .results.arc_challenge_mt_it["acc_norm,none"] // .results.arc_challenge_it["acc_norm,none"]' 100
print_metric " arc_challenge_nl: " "arc_challenge_nl" '.results.arc_challenge_nl.acc_norm // .results.arc_challenge_mt_nl["acc_norm,none"] // .results.arc_challenge_nl["acc_norm,none"]' 100
print_metric " arc_challenge_pl: " "arc_challenge_pl" '.results.arc_challenge_pl.acc_norm // .results.arc_challenge_mt_pl["acc_norm,none"] // .results.arc_challenge_pl["acc_norm,none"]' 100
print_metric " arc_challenge_pt: " "arc_challenge_pt" '.results.arc_challenge_pt.acc_norm // .results.arc_challenge_mt_pt["acc_norm,none"] // .results.arc_challenge_pt["acc_norm,none"]' 100

echo
print_metric " arc_challenge_bg: " "arc_challenge_bg" '.results.arc_challenge_bg.acc_norm // .results.arc_challenge_mt_bg["acc_norm,none"] // .results.arc_challenge_bg["acc_norm,none"]' 100
print_metric " arc_challenge_cs: " "arc_challenge_cs" '.results.arc_challenge_cs.acc_norm // .results.arc_challenge_mt_cs["acc_norm,none"] // .results.arc_challenge_cs["acc_norm,none"]' 100
print_metric " arc_challenge_et: " "arc_challenge_et" '.results.arc_challenge_et.acc_norm // .results.arc_challenge_mt_et["acc_norm,none"] // .results.arc_challenge_et["acc_norm,none"]' 100
print_metric " arc_challenge_lt: " "arc_challenge_lt" '.results.arc_challenge_lt.acc_norm // .results.arc_challenge_mt_lt["acc_norm,none"] // .results.arc_challenge_lt["acc_norm,none"]' 100
print_metric " arc_challenge_lv: " "arc_challenge_lv" '.results.arc_challenge_lv.acc_norm // .results.arc_challenge_mt_lv["acc_norm,none"] // .results.arc_challenge_lv["acc_norm,none"]' 100
print_metric " arc_challenge_ro: " "arc_challenge_ro" '.results.arc_challenge_ro.acc_norm // .results.arc_challenge_mt_ro["acc_norm,none"] // .results.arc_challenge_ro["acc_norm,none"]' 100
print_metric " arc_challenge_sk: " "arc_challenge_sk" '.results.arc_challenge_sk.acc_norm // .results.arc_challenge_mt_sk["acc_norm,none"] // .results.arc_challenge_sk["acc_norm,none"]' 100
print_metric " arc_challenge_sl: " "arc_challenge_sl" '.results.arc_challenge_sl.acc_norm // .results.arc_challenge_mt_sl["acc_norm,none"] // .results.arc_challenge_sl["acc_norm,none"]' 100

echo
# --- code tasks ---
print_metric " humaneval-unstripped_pass@1: " "humaneval-unstripped_pass@1" '."humaneval-unstripped"."pass@1"' 100
print_metric "humaneval-unstripped_pass@10: " "humaneval-unstripped_pass@10" '."humaneval-unstripped"."pass@10"' 100
print_metric " humaneval_pass@1: " "humaneval_pass@1" '.humaneval."pass@1"' 100
print_metric "humaneval_pass@10: " "humaneval_pass@10" '.humaneval."pass@10"' 100
print_metric "      mbpp_pass@1: " "mbpp_pass@1" '.mbpp."pass@1"' 100
print_metric "     mbpp_pass@10: " "mbpp_pass@10" '.mbpp."pass@10"' 100

echo

# --- misc QA / reasoning ---
print_metric "         arc_easy: " "arc_easy" '.results.arc_easy.acc // .results.arc_easy["acc,none"]' 100
print_metric "            boolq: " "boolq" '.results.boolq.acc // .results.boolq["acc,none"]' 100
print_metric "          nq_open: " "nq_open" '.results.nq_open.em // .results.nq_open["exact_match,remove_whitespace"]' 100
print_metric " openbookqa (acc): " "openbookqa" '.results.openbookqa.acc // .results.openbookqa["acc,none"]' 100
print_metric "       piqa (acc): " "piqa" '.results.piqa.acc // .results.piqa["acc,none"]' 100
print_metric "      squad2 (f1): " "squad2" '.results.squad2.f1 // .results.squadv2["f1,none"]' 1
print_metric "         triviaqa: " "triviaqa" '.results.triviaqa.em // .results.triviaqa["exact_match,remove_whitespace"]' 100

echo

# --- safety / other ---
print_metric "    toxigen (acc): " "toxigen" '.results.toxigen.acc // .results.toxigen["acc,none"]' 100

# Optional extras (print only if present; also support vllm/timestamped)
bele_file="$(pick_file "belebele_fin")"
if [ -n "$bele_file" ]; then
    echo
    echo -n "     belebele_fin: "
    jq -r '.results.belebele_fin_Latn["acc,none"]' "$bele_file" 2>/dev/null | awk '{print $1 * 100}'
fi

ifeval_file="$(pick_file "ifeval")"
if [ -n "$ifeval_file" ]; then
    echo
    echo -n "     ifeval prompt_loose: "
    jq -r '.results.ifeval["prompt_level_loose_acc,none"]' "$ifeval_file" 2>/dev/null | awk '{print $1 * 100}'
fi

ifeval_fi_file="$(pick_file "ifeval_fi")"
if [ -n "$ifeval_fi_file" ]; then
    echo -n "     ifeval_fi prompt_loose: "
    jq -r '.results.ifeval_fi["prompt_level_loose_acc,none"]' "$ifeval_fi_file" 2>/dev/null | awk '{print $1 * 100}'
fi

multi_if_file="$(pick_file "multi_if")"
if [ -n "$multi_if_file" ]; then
    echo -n "     multi_if english_average: "
    jq -r '.results.english_average' "$multi_if_file" 2>/dev/null
fi

mtbench_en_file="$(pick_file "mtbench_judge_en")"
if [ -n "$mtbench_en_file" ]; then
    echo -n "     mtbench_judge_en first_turn_score: "
    jq -r '.results.first_turn_score' "$mtbench_en_file" 2>/dev/null
fi

mtbench_fi_file="$(pick_file "mtbench_judge_fi")"
if [ -n "$mtbench_fi_file" ]; then
    echo -n "     mtbench_judge_fi first_turn_score: "
    jq -r '.results.first_turn_score' "$mtbench_fi_file" 2>/dev/null
fi

if [ -n "$mtbench_en_file" ]; then
    echo -n "     mtbench_judge_en average_score: "
    jq -r '.results.average_score' "$mtbench_en_file" 2>/dev/null
fi

if [ -n "$mtbench_fi_file" ]; then
    echo -n "     mtbench_judge_fi average_score: "
    jq -r '.results.average_score' "$mtbench_fi_file" 2>/dev/null
fi

alpaca_en_file="$(pick_file "alpaca_eval_en")"
if [ -n "$alpaca_en_file" ]; then
    echo -n "     alpaca_eval_en length_controlled_winrate: "
    jq -r '.results.length_controlled_winrate' "$alpaca_en_file" 2>/dev/null
fi

alpaca_fi_file="$(pick_file "alpaca_eval_fi")"
if [ -n "$alpaca_fi_file" ]; then
    echo -n "     alpaca_eval_fi length_controlled_winrate: "
    jq -r '.results.length_controlled_winrate' "$alpaca_fi_file" 2>/dev/null
fi

echo
