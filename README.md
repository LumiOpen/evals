# evals

This is a script to simplify running many evals simultaneously in our slurm
environment.  It accepts hugging face model IDs (i.e. org/modelname) or
directory paths to models in the huggingface format.


Set the active LUMI billing project before submitting jobs. The launcher no
longer defaults to a historical project:

```bash
export LUMI_PROJECT=project_123456789
```

For partial-node GPU jobs, request CPU and memory in proportion to the GPU
allocation instead of reserving the whole node:

```bash
python main.py arc_easy \
    --model org/model \
    --backend vllm \
    --partition dev-g \
    --gres gpu:mi250:1 \
    --cpus-per-task 7 \
    --mem 60G \
    --time 00:10:00
```

Generated sbatch scripts are kept under `--work_dir`, and successful
submissions print and record a JSON receipt containing the parsable Slurm job
ID and the effective resource/runtime settings. Temporary container state is
isolated by Slurm job ID, so multiple evaluations may safely share a node.

`HF_TOKEN` is not forwarded into containers by default. Use
`--forward-hf-token` only for models that require authentication.

## Usage

Common run configs can be found in command line scripts like `cpt.sh` which you
can run like

```bash
# run common tests for finnish CPT evals
sh cpt.sh /path/to/somemodel
```

For vLLM evaluations, there's also `cpt-vllm.sh`:

```bash
# run vLLM tests for finnish CPT evals
sh cpt-vllm.sh org/modelname
```

You can also invoke the script directly to run individual evals as needed.

```bash
python main.py \
    --partition standard-g
    --time 04:00:00
    --model path/to/model_step1234 \
    eval_name1 eval_name2
```

### Backend selection

The script supports different backends:

```bash
# HuggingFace backend (default)
python main.py arc_challenge --model path/to/model

# vLLM backend for faster inference
python main.py arc_challenge --model path/to/model --backend vllm

# Dummy backend for cache-only runs (no model loading)
python main.py arc_challenge --model path/to/model --backend dummy \
    --lm_eval_args "--use_cache /path/to/cache.db"

# Custom model arguments (works with all backends)
python main.py arc_challenge --model path/to/model --backend vllm \
    --model_args "max_model_len=8192,gpu_memory_utilization=0.95"
```

**Note:** The vLLM backend is experimental. Performance and correctness have not been confirmed to be comparable with the HuggingFace backend.

### Custom lm-evaluation-harness source

You can specify a custom lm-evaluation-harness source (works with all backends):

```bash
# Use a different GitHub repository
python main.py arc_challenge --model path/to/model \
    --lm_eval https://github.com/user/custom-lm-eval.git

# Use a specific branch or tag
python main.py arc_challenge --model path/to/model \
    --lm_eval https://github.com/LumiOpen/lm-evaluation-harness@feature-branch

# Use local development version
python main.py arc_challenge --model path/to/model \
    --lm_eval /path/to/local/lm-evaluation-harness
```

Refs may be branch names, tags, or exact commit SHAs. Exact commits are fetched
directly and checked out detached. Pin a reviewed commit for reproducible runs.

### Testing with limited examples

For testing purposes, you can limit the number of examples per task:

```bash
python main.py arc_challenge --model path/to/model --limit 50
```

The script will try to avoid running nevals for which you already have results
or for which there already appear to be jobs in the slurm queue.  It determines
this latter case by reviewing the logs in `command_history.jsonl`.

Slurm job output is stored in the `logs` subdir.

## Results

Output is written by default into the `output` subdirectory.  The results are
stored in json format which is not particularly convenient.  There is
a `summary.sh` script which will extract the correct scores each eval that is
available.

```bash
sh summary.sh output/v2/meta-llama/Llama-3.1-8B
```

## Watching job status.

The `watch.py` script is a convenience script to help keep track of the jobs you
have running, it has two operational modes.

In the default mode it prints the jobs that are currently in the queue or
running, and if available it prints the last line in the error log in each
file, which often contains the most recent tqdm progress bar for jobs that are
running.  If you specify the `--once` flag it will do this and exit, if you do
not specify the `--once` flag it will keep checking job status periodically and
provide updates as jobs complete.

```bash
$ python watch.py --once
9678732 /scratch/project_462000353/converted-checkpoints/llama31_8B_culturax50B_2e-5/iter_0011920 hellaswag_mt_fi is queued.
9678731 /scratch/project_462000353/converted-checkpoints/llama31_8B_culturax50B_2e-5/iter_0011920 hellaswag is queued.
9678730 /scratch/project_462000353/converted-checkpoints/llama31_8B_culturax50B_2e-5/iter_0011920 gsm8k_mt_fi is queued.
9678729 /scratch/project_462000353/converted-checkpoints/llama31_8B_culturax50B_2e-5/iter_0011920 gsm8k is queued.
9678728 /scratch/project_462000353/converted-checkpoints/llama31_8B_culturax50B_2e-5/iter_0011920 mmlu_mt_fi is queued.
9670938 meta-llama/Llama-3.1-70B hellaswag_mt_fi is running.
Running loglikelihood requests:  43%|████▎     | 17337/40168 [13:41:50<16:37:51,  2.62s/it]
9678481 /scratch/project_462000353/converted-checkpoints/llama31-8b-tp2-pp1-megatron-format-lr5e-5_iter_0011920_bfloat16 gsm8k is running.
Running generate_until requests:   4%|▍         | 59/1319 [07:29<1:57:40, 5.60s/it]
9678727 /scratch/project_462000353/converted-checkpoints/llama31_8B_culturax50B_2e-5/iter_0011920 mmlu is running.
Running loglikelihood requests:  83%|████████▎ | 46725/56168 [34:57<03:33, 44.33it/s]
```

There is another new operational mode recently added that might be more useful.
Specifying the `--hist` flag will show a report of the jobs that have completed
in the last 3 days (controllable with the `--days` flag) sorted by model name
and status.  It also does some coalescing: if an eval is ultimately successful,
it won't bother reporting on failed runs, etc.  This is helpful to identify
evals which have failed and need to be investigated or rerun.

```bash
$ python watch.py --hist --days 1
Model: meta-llama/Llama-3.1-70B
Results dir: /pfs/lustrep2/scratch/project_462000353/jburdge/git/evals/output/v2/meta-llama/Llama-3.1-70B
Completed:
    gsm8k_mt_fi
    gsm8k_mt_fi
Running/Queued:
    hellaswag_mt_fi 9670938
Failed:
    hellaswag_mt_fi /pfs/lustrep2/scratch/project_462000353/jburdge/git/evals/logs/9639428.err
```

## Command History

All evals are logged in `command_history.jsonl`, which is used by various
scripts to monitor job status and report history.

An entry looks like this.

```bash
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

You can utilize the information directly as well. If you've just
queued up a bunch of evals against a model and realized you made a mistake and
need to cancel them all, you could do something like this to save a lot of
typing:

```bash
grep /path/to/model command_history.jsonl | jq -r .job_id | xargs scancel
```