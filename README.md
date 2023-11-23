# evals

This is a script to simplify running many evals simultaneously in our slurm environment.

## usage

```bash
python main.py \
    --model path/to/model_step1234 \
    --tokenizer path/to/model_step1234
    eval_name
```

Commands will be logged to `command_history.jsonl` to help you look up job_ids and see which evals have been run.

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

You can now specify multiple evals with a single command:

```
python main.py --model foo --tokenizer foo arc_challenge hellaswag mmlu truthfulqa_mc winogrande gsm8k drop finbench
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
cat output/poro-34b/step48672/finbench.json | jq -r '.results | to_entries | .[] | [.key, .value.multiple_choice_grade] | @csv'
```


