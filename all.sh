echo $1
sleep 1
python main.py --partition standard-g --model $1 arc_challenge hellaswag mmlu truthfulqa_mc winogrande gsm8k humaneval_pass@1 humaneval_pass@10 mbpp_pass@1 mbpp_pass@10 arc_easy boolq nq_open openbookqa piqa triviaqa toxigen squad2 finbench_3shot
