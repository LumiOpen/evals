#!/bin/bash

echo 'This script will rewrite summaries/, latex_tables/, and figures/. Continue? (y/n)'
while true; do
    read answer
    case "$answer" in
	[yY]) break;;
	[nN]) exit;;
	*) echo 'Please answer "y" or "n"'
    esac
done

rm -rf summaries
mkdir summaries

rm -rf latex_tables
mkdir latex_tables

rm -rf figures
mkdir figures

set -euo pipefail

### FIN-bench

# FIN-bench TSV summaries
mkdir summaries/finbench
for f in output/{poro-34b,TurkuNLP}/*/finbench.json; do
    d=$(dirname "$f")
    m=${d#output/}
    m=${m//\//_}
    o=summaries/finbench/${m}.tsv
    python3 finbench_summary.py "$f" > "$o"
done

# FIN-bench detailed Poro results table
o=latex_tables/poro-finbench-progression.tex

data=$(ls summaries/finbench/poro-34b_step*.tsv \
	   | perl -pe 's/(.*step(\d+))/$2:$1/' | sort -n | tr '\n' ' ')

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
	summaries/finbench/poro-34b_step* \
	summaries/finbench/TurkuNLP_gpt3-finnish-8B.tsv \
	summaries/finbench/TurkuNLP_bloom-finnish-176b.tsv
