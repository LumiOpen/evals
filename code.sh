QUEUE=standard-g
# we define shorter time periods to allow evals to get scheduled more quickly
# via backfill when the cluster is full.
# times are based on a 70B model
# don't try to cut it too close, allow 1-2 hours additional time.
python main.py --time 12:00:00 --partition $QUEUE --model $1 humaneval_pass@1 mbpp_pass@1
python main.py --time 24:00:00 --partition $QUEUE --model $1 humaneval_pass@10
python main.py --time 48:00:00 --partition $QUEUE --model $1 mbpp_pass@10
