# evals
This is a script to simplify running many evals simultaneously in our slurm environment.
## Installation
- these should be done and and modify the `templates/` according the changes
- without the modifications evals may crash due depency issues caused by local instals &rarr; in case of failure just run single task to get working venv
1. clone the repos and link output
```bash
git clone https://github.com/LumiOpen/evals
git clone https://github.com/LumiOpen/evaluation-internal
ln -s evaluation-internal evals/output
cd evals 
```
2. Set up environment variables (TODO venvs):
- lm-eval-harness2 related `$PYTHONUSERBASE` to `scratch`
- lm-eval-harness related `$PYTHONUSERBASE` to `scratch`
- finbench related `$PYTHONUSERBASE` to `scratch`
- bigcode related `$PYTHONUSERBASE` to `scratch`
- modify each template in `templates` so that you either specify pythonuserbase or PYTHONPATH to these folders
3. venv for conversion script
- include `transformers==4.37.2`
- currently loads from `$PYTHONUSERBASE`
## Running the jobs
1. convert a checkpoint
    - conversion scripts can be found from `convert-scripts`
    - based on our Megatron-LM fork and conversion is dependent on Megatron-LM 
    ```bash
    sbatch convert_europa_7B-sbatch.sh path/to/megatron/checkpoint
    ```
    - conversion is not working corretly &rarr; change manually `num_key_value_heads:1` to 32 in converted checkpoint
2. run single eval
```bash
module load cray-python
python python main.py --trust_remote_code --model /path/to/converted/ --partition dev-g --time 01:00:00 toxigen
```
- evals fails from time to time due unstable environments &rarr; if it fails, just run `main.py` to run single task succesfully to restore the environment
- Commands will be logged to `command_history.jsonl` to help you look up job_ids and see which evals have been run.
```
{
    "timestamp": "2023-11-09 08:21:12",
    "script_name": "/tmp/tmpvv66ri7g",
    "job_id": "4868114",
    "eval": "hellaswag",
    "model": "/scratch/project_462000319/general-tools/checkpoints/33B_torch_step70128_bfloat16",
    "tokenizer": "/scratch/project_462000319/tokenizers/tokenizer_v6_fixed_fin",
    "err_log": "/pfs/lustrep4/scratch/project_462000319/evals/logs/4868114.err",
    "out_log": "/pfs/lustrep4/scratch/project_462000319/evals/logs/4868114.out",
    "output_file": "/pfs/lustrep4/scratch/project_462000319/evals/output/poro-34b/step70128/hellaswag.json"
}

```
- You can specify multiple evals with a single command:

```
python main.py --model foo --tokenizer foo arc_challenge hellaswag mmlu truthfulqa_mc winogrande gsm8k finbench_3shot
```

If you are running a larger model and need more GPUs, you can specify the --gre
argument to override the default (2):

```
python main.py --model foo --gre gpu:mi250:4 eval1 eval2
```
3. Run all evals
```bash
module load cray-python
bash all.sh /path/to/converted/checkpoint/
```
4. Gather summary of results
```bash

bash summary.sh /output/v2/your_model/your_iter
```
## Watching

You can use the included watch.py to monitor squeue to catch when jobs
complete. It will monitor job status and show job results.

## Totaling

Some tests require averaging various scores together.  Here's a sample script to
average mmlu.

```
cat output/mistralai/Mistral-7B-v0.1/mmlu.json | jq '.results | .[] | .acc' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'
```

## Exporting

To export a run's results as csv, you can use a command like the following:

```
cat output/poro-34b/step48672/finbench_3shot.json | jq -r '.results | to_entries | .[] | [.key, .value.multiple_choice_grade] | @csv'
```


