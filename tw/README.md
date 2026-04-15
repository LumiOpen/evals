# TensorWave Long-Context Evaluations

HELMET (128K) and LongPPL evaluation scripts for the TensorWave MI325X cluster.

## Setup

Run once from a node with `/shared_silo/scratch` access:

```bash
ssh tus1-vm-amd-misc-05
cd /path/to/evals/tw
bash setup.sh
```

This clones HELMET and the LumiOpen lm-eval fork, and downloads HELMET data (~34 GB).

By default everything goes under `/shared_silo/scratch/$USER/`. Override with `SCRATCH`:

```bash
SCRATCH=/shared_silo/scratch/ezosa bash setup.sh
```

## HELMET

Runs all 7 HELMET categories (recall, rag, longqa, summ, icl, rerank, cite) at 128K context with vLLM on 8 GPUs.

```bash
# Local model path
sbatch --export=ALL,MODEL=/shared_silo/scratch/models/Meta-Llama-3.1-8B eval_helmet.sbatch

# HuggingFace model
sbatch --export=ALL,MODEL=LumiOpen/Llama-Poro-2-Long-Base eval_helmet.sbatch

# Run specific configs only
sbatch --export=ALL,MODEL=...,CONFIGS='rag.yaml recall.yaml' eval_helmet.sbatch
```

Base models are auto-detected (no `chat`/`instruct`/`it` suffix) and run without chat templates. Override with `OPTIONS`.

Results go to `$SCRATCH/HELMET/output/<model_name>/`.

## LongPPL

Cross-lingual key-token perplexity (English + Finnish) on 1 GPU.

```bash
# Standard model
sbatch --export=ALL,MODEL=/shared_silo/scratch/models/Meta-Llama-3.1-8B eval_longppl.sbatch

# Model needing RoPE scaling (e.g. extending 8K → 32K)
sbatch --export=ALL,MODEL=LumiOpen/Poro-2-8B-base,ROPE_SCALING=true eval_longppl.sbatch
```

Results go to `$SCRATCH/eval_results/longppl/`.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | (required) | Model path or HF ID |
| `SCRATCH` | `/shared_silo/scratch/$USER` | Scratch root directory |
| `IMG` | `rocm_vllm_rocm7.0.0_vllm_0.11.2_20251210.sif` | Container image |
| `CONFIGS` | all 7 HELMET configs | Space-separated config list (HELMET only) |
| `TAG` | `v1` | Output tag (HELMET only) |
| `SEED` | `42` | Random seed (HELMET only) |
| `ROPE_SCALING` | `false` | Enable RoPE scaling (LongPPL only) |
| `TASKS` | `crosslingual_longppl_en,crosslingual_longppl_fi` | lm-eval tasks (LongPPL only) |

## Container

Both scripts use the ROCm vLLM container at `/shared_silo/scratch/containers/`. The container needs `ulimit -n 65536` for HELMET's multiprocessing data loader — this is already set in the script.

Dependencies (`pytrec_eval`, `rouge_score`, etc.) are pip-installed into per-job temp dirs at runtime to avoid cross-job interference.
