QUEUE=standard-g
PROJECT=project_462000615
# we decine shorter time periods to allow evals to get scheduled more quickly
# via backfill when the cluster is full.
# times are based on a 70B model
# don't try to cut it too close, allow 1-2 hours additional time.
python main.py --project $PROJECT --time 04:00:00 --partition $QUEUE --model $1 arc_challenge_mt_fi arc_challenge_mt_da arc_challenge_mt_is arc_challenge_mt_nb arc_challenge_mt_sv arc_challenge truthfulqa_mc truthfulqa_mc_mt_fi winogrande arc_easy boolq nq_open openbookqa piqa toxigen 
python main.py --project $PROJECT --time 04:00:00 --partition $QUEUE --model $1 arc_challenge_mt_de arc_challenge_mt_el arc_challenge_mt_es arc_challenge_mt_hu arc_challenge_mt_it arc_challenge_mt_pl arc_challenge_mt_pt arc_challenge_mt_fr arc_challenge_mt_nl arc_challenge_mt_bg arc_challenge_mt_cs arc_challenge_mt_et arc_challenge_mt_lt arc_challenge_mt_lv arc_challenge_mt_ro arc_challenge_mt_sk arc_challenge_mt_sl
python main.py --project $PROJECT --time 10:00:00 --partition $QUEUE --model $1 mmlu mmlu_mt_fi
python main.py --project $PROJECT --time 24:00:00 --partition $QUEUE --model $1 hellaswag hellaswag_mt_fi gsm8k gsm8k_mt_fi triviaqa
# squad2 is broken right now

# finbench is broken
#python main.py --time 06:00:00 --partition $QUEUE --model $1 finbench_3shot
