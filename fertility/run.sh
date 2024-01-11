#!/bin/bash

model_langs=(
    "gpt2:eng"
    "bigscience/bloom:fra"
    "bigscience/bloom:deu"
)

for model_lang in "${model_langs[@]}"; do
    model="${model_lang%%:*}"
    lang="${model_lang#*:}"
    result=$(
	python3 fertility.py $model flores200_dataset/dev/$lang* 2>/dev/null \
	    | perl -pe 's/^fertility\s+//; s/ /\t/'
    )
    echo "$model"$'\t'"$lang"$'\t'"$result"
done
    
#python3 fertility.py gpt2 flores200_dataset/dev/eng_Latn.dev 2>/dev/null

