# evals

This is a script to simplify running many evals simultaneously in our slurm environment.

## usage

```bash
python main.py \
    --model path/to/model_step1234 \
    --tokenizer path/to/model_step1234
    --output_dir results/modelname_step1234
    eval_name
```
An output_dir is technically optional, but you should probably always use it to group results together for
each model and checkpoint so that you can

Commands will be logged to `command_history.jsonl` to help you look up job_ids and see which evals have been run.

```
jq 'select(.model == "path/to/model_step1234")' command_history.jsonl 
```