#!/bin/bash
#SBATCH --job-name=tester
#SBATCH --account=project_2007628
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

##SBATCH --partition=gpusmall
##SBATCH --time=4:00:00 # run time d-hh:mm:ss
##SBATCH --gres=gpu:a100:2

#SBATCH --partition=gputest
#SBATCH --time=0:15:00 # run time d-hh:mm:ss
#SBATCH --gres=gpu:a100:2


module load pytorch
source /scratch/project_2007628/jburdge/venv/bin/activate

echo "$(python -c 'import torch; print(torch.cuda.is_available())')"

cd /scratch/project_2007628/jburdge/git/finbench

#export MODEL=/scratch/project_2007628/models/33B_torch_step70128_bfloat16
export MODEL=/scratch/project_2007628/zosaelai2/models/sft_finetuned/merged-poro-oasst-en-lora
export TOKENIZER=/scratch/project_2007628/tokenizers/tokenizer_v6_fixed_fin

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL,use_accelerate=True,tokenizer=$TOKENIZER,dtype=bfloat16\
    --device cuda:0 \
    --no_cache \
    --tasks bigbench_1_digit_addition,bigbench_1_digit_division,bigbench_1_digit_multiplication,bigbench_1_digit_subtraction,bigbench_2_digit_addition,bigbench_2_digit_division,bigbench_2_digit_multiplication,bigbench_2_digit_subtraction,bigbench_3_digit_addition,bigbench_3_digit_division,bigbench_3_digit_multiplication,bigbench_3_digit_subtraction,bigbench_4_digit_addition,bigbench_4_digit_division,bigbench_4_digit_multiplication,bigbench_4_digit_subtraction,bigbench_5_digit_addition,bigbench_5_digit_division,bigbench_5_digit_multiplication,bigbench_5_digit_subtraction,bigbench_analogies,bigbench_emotions,bigbench_empirical_judgments,bigbench_general_knowledge,bigbench_harmless,bigbench_helpful,bigbench_honest,bigbench_intent_recognition,bigbench_misconceptions,bigbench_one_sentence,bigbench_one_sentence_no_prompt,bigbench_other,bigbench_paraphrase,bigbench_sentence_ambiguity,bigbench_similarities_abstraction,bigbench_two_sentences \
    --num_fewshot 3 \
    --output_path output/results-bigbench-finnish-batch.json
