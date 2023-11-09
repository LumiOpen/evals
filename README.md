# evals

This is a script to simplify running many evals simultaneously in our slurm environment.

## usage

```bash
python main.py \
    --model path/to/model_step1234 \
    --tokenizer path/to/model_step1234
    --output_dir results/model_step1234
    eval_name
```
An output_dir is technically optional, but you should probably always use it to group results together for
each model and checkpoint so that you can find them.

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
