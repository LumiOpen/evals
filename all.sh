python main.py --time 04:00:00 --partition standard-g --model $1 finbench_3shot arc_challenge_fi arc_challenge_da arc_challenge_nb arc_challenge_sv arc_challenge truthfulqa_mc winogrande  humaneval_pass@1 mbpp_pass@1 arc_easy boolq nq_open openbookqa piqa triviaqa toxigen 
python main.py --time 24:00:00 --partition standard-g --model $1 mmlu humaneval_pass@10 mbpp_pass@10 squad2 hellaswag gsm8k
