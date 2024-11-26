QUEUE=standard-g
# we decine shorter time periods to allow evals to get scheduled more quickly
# via backfill when the cluster is full.
# times are based on a 70B model
# don't try to cut it too close, allow 1-2 hours additional time.
python main.py --time 04:00:00 --partition $QUEUE --model $1 arc_challenge_mt_fi arc_challenge_mt_da arc_challenge_mt_is arc_challenge_mt_nb arc_challenge_mt_sv arc_challenge truthfulqa_mc winogrande arc_easy boolq nq_open openbookqa piqa toxigen 
python main.py --time 04:00:00 --partition $QUEUE --model $1 arc_challenge_mt_de arc_challenge_mt_el arc_challenge_mt_es arc_challenge_mt_hu arc_challenge_mt_it arc_challenge_mt_pl arc_challenge_mt_pt arc_challenge_mt_fr arc_challenge_mt_nl
python main.py --time 03:00:00 --partition $QUEUE --model $1 finbench_3shot
python main.py --time 10:00:00 --partition $QUEUE --model $1 mmlu
#skipping squad as its not working
#python main.py --time 24:00:00 --partition $QUEUE --model $1 squad2 hellaswag gsm8k triviaqa
python main.py --time 07:00:00 --partition $QUEUE --model $1 hellaswag gsm8k triviaqa


