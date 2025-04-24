QUEUE=standard-g
PROJECT=project_462000615

# this can go very long in models that are not good at this language, i guess?
python3 main.py --time 03:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 --force flores200_trans_en_fi flores200_trans_fi_en
python3 main.py --time 12:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 --force arc_challenge arc_challenge_mt_fi truthfulqa_mc truthfulqa_mc_mt_fi
python3 main.py --time 10:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 --force mmlu_mt_fi
python3 main.py --time 48:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 --force hellaswag_mt_fi
python3 main.py --time 14:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 --force mmlu
python3 main.py --time 12:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 --force gsm8k gsm8k_mt_fi
python3 main.py --time 14:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 --force ifeval ifeval_fi