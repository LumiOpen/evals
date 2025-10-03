QUEUE=standard-g
PROJECT=project_462000963

# this can go very long in models that are not good at this language, i guess?
python3 main.py --time 3:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 flores200_trans_en_fi flores200_trans_fi_en
python3 main.py --time 18:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 arc_challenge arc_challenge_mt_fi truthfulqa_mc truthfulqa_mc_mt_fi
python3 main.py --time 12:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 mmlu_mt_fi
python3 main.py --time 48:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 hellaswag_mt_fi
python3 main.py --time 48:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 hellaswag
python3 main.py --time 04:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 goldenswag
python3 main.py --time 04:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --model $1 goldenswag_0shot
python3 main.py --time 12:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 mmlu
python3 main.py --time 12:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 gsm8k gsm8k_mt_fi
python3 main.py --time 20:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 ifeval
python3 main.py --time 24:00:00 --project $PROJECT --partition $QUEUE --apply_chat_template --fewshot_as_multiturn --model $1 ifeval_fi
