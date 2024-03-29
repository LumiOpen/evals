#!/bin/bash

# Mapping between model names and labels
declare -A NAME_MAP=(
    [Poro-34B]='Poro 34B'
    [gpt3-finnish-13B]='FinGPT 13B'
    [gpt3-finnish-8B]='FinGPT 8B'
    [falcon-40b]='Falcon 40B'
    [llama-30b]='Llama 33B'
    [mpt-30b]='MPT 30B'
    [starcoderbase]='StarCoder'
)

# Order of models in tables
ORDER_STR='Poro 34B,Llama 33B,MPT 30B,Falcon 40B,FinGPT 8B,FinGPT 13B,StarCoder'

# Format NAME_MAP as "key:value,key:value,..."
NAME_MAP_STR=""
for key in "${!NAME_MAP[@]}"; do
    value=${NAME_MAP[$key]}
    NAME_MAP_STR+="$key:$value,"
done
NAME_MAP_STR=${NAME_MAP_STR%,}    # trailing ","

echo 'This script will rewrite summaries/, latex_tables/, and figures/. Continue? (y/n)'
while true; do
    read answer
    case "$answer" in
	[yY]) break;;
	[nN]) exit;;
	*) echo 'Please answer "y" or "n"'
    esac
done

for d in summaries latex_tables figures; do
    rm -rf "$d"
    mkdir "$d"
done

set -euo pipefail

### FIN-bench

# FIN-bench TSV summaries
mkdir summaries/finbench
for f in output/{poro-34b,TurkuNLP}/*/finbench_3shot.json; do
    d=$(dirname "$f")
    m=${d#output/}
    m=${m//\//_}
    o=summaries/finbench/${m}.tsv
    python3 finbench_summary.py "$f" > "$o"
done

# FIN-bench detailed Poro results table
o=latex_tables/poro-finbench-progression.tex

data=$(ls summaries/finbench/poro-34b_step{25632,48672,70128,95328,119232,143712,166752,190656,214560,238418}.tsv \
	   | perl -pe 's/(.*step(\d+))/$2:$1/' \
	   | sort -n | tr '\n' ' ')

python3 make_latex_table.py \
	--caption 'Poro progression on FIN-bench' \
	--label 'tab:poro-finbench-progression' \
	--decimals 1 \
	--percentage \
	--wide \
	$data \
	> "$o"

# FIN-bench Poro progression
python3 plot_poro_progression.py \
	--output figures/poro-finbench-progression.pdf \
	summaries/finbench/poro-34b_step{25632,48672,70128,95328,119232,143712,166752,190656,214560,238418}.tsv \
	summaries/finbench/TurkuNLP_gpt3-finnish-8B.tsv \
	summaries/finbench/TurkuNLP_bloom-finnish-176b.tsv

### Perplexity

# Character-level perplexity TSV summaries
mkdir summaries/char-ppl
for f in perplexity/results/*.txt; do
    egrep 'mean cppl:' $f \
	| perl -pe 's/^fin.*: /Finnish\t/' \
	| perl -pe 's/^eng.*: /English\t/' \
	| perl -pe 's/^code.*: /Code\t/' \
	| perl -pe 's/^overall.*: /Average\t/' \
	       > summaries/char-ppl/$(basename $f .txt).tsv
done

# Character-level perplexity table
o=latex_tables/char-ppl.tex

data=$(ls summaries/char-ppl/*.tsv \
	   | perl -pe 's/(.*\/(.*)\.tsv)/$2:$1/' | sort -n | tr '\n' ' ')

echo "$data"

python3 make_latex_table.py \
	--caption 'Character-level perplexity' \
	--label 'tab:char-ppl' \
	--decimals 3 \
	--name-map "$NAME_MAP_STR" \
	--order "$ORDER_STR" \
	$data \
	> "$o"
