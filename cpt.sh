QUEUE=standard-g
PROJECT=project_462000353

# this can go very long in models that are not good at this language, i guess?
python main.py --time 12:00:00 --project $PROJECT --partition $QUEUE --model $1 flores200_trans_en_fi flores200_trans_fi_en
python main.py --time 10:00:00 --project $PROJECT --partition $QUEUE --model $1 arc_challenge arc_challenge_mt_fi truthfulqa_mc truthfulqa_mc_mt_fi goldenswag_mt_fi
python main.py --time 10:00:00 --project $PROJECT --partition $QUEUE --model $1 mmlu mmlu_mt_fi
python main.py --time 48:00:00 --project $PROJECT --partition $QUEUE --model $1 gsm8k gsm8k_mt_fi hellaswag hellaswag_mt_fi goldenswag
