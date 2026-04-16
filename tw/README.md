# TensorWave Long-Context Evaluations

HELMET (8K–128K) and LongPPL evaluation scripts for the TensorWave MI325X cluster.

## Setup

Already done — HELMET code + data and the lm-eval fork are at `/shared_silo/scratch/shared/`. If you need to re-run setup (e.g. after a data wipe):

```bash
ssh tus1-vm-amd-misc-05
bash setup.sh
```

## HELMET

Runs all 8 HELMET categories (recall, rag, longqa, summ, icl, rerank, cite, alce_nocite) at all context lengths (8k, 16k, 32k, 64k, 128k) with vLLM on 8 GPUs.

```bash
# Full sweep (8k–128k, all categories) — default
sbatch --export=ALL,MODEL=/shared_silo/scratch/models/Meta-Llama-3.1-8B eval_helmet.sbatch

# HuggingFace model
sbatch --export=ALL,MODEL=LumiOpen/Llama-Poro-2-Long-Base eval_helmet.sbatch

# 128k only (faster, ~4h)
sbatch --export=ALL,MODEL=...,CONFIGS='recall.yaml rag.yaml longqa.yaml summ.yaml icl.yaml rerank.yaml cite.yaml' eval_helmet.sbatch

# Specific configs
sbatch --export=ALL,MODEL=...,CONFIGS='rag_short.yaml rag.yaml' eval_helmet.sbatch
```

Base models are auto-detected (no `chat`/`instruct`/`sft`/`it` suffix) and run without chat templates.

Results go to `/shared_silo/scratch/$USER/helmet_output/<model_name>/`.

## LongPPL

Cross-lingual key-token perplexity (English + Finnish) on 1 GPU.

```bash
# Standard model
sbatch --export=ALL,MODEL=/shared_silo/scratch/models/Meta-Llama-3.1-8B eval_longppl.sbatch

# Model needing RoPE scaling (e.g. extending 8K → 32K)
sbatch --export=ALL,MODEL=LumiOpen/Poro-2-8B-base,ROPE_SCALING=true eval_longppl.sbatch
```

Results go to `/shared_silo/scratch/$USER/eval_results/longppl/`.

## Shared vs per-user paths

| Path | What |
|------|------|
| `/shared_silo/scratch/shared/HELMET/` | HELMET code + data (shared, read-only at runtime) |
| `/shared_silo/scratch/shared/lm-eval-longppl/` | LumiOpen lm-eval fork (shared) |
| `/shared_silo/scratch/$USER/helmet_output/` | Your HELMET results |
| `/shared_silo/scratch/$USER/eval_results/longppl/` | Your LongPPL results |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | (required) | Model path or HF ID |
| `SHARED` | `/shared_silo/scratch/shared` | Shared data root |
| `SCRATCH` | `/shared_silo/scratch/$USER` | Per-user scratch for outputs |
| `IMG` | `rocm_vllm_rocm7.0.0_vllm_0.11.2_20251210.sif` | Container image |
| `CONFIGS` | all 8 categories × short+128k | Space-separated config list (HELMET only) |
| `CONFIG_DIR` | `$HELMET_DIR/configs` | Path to HELMET config directory |
| `TAG` | `v1` | Output tag (HELMET only) |
| `SEED` | `42` | Random seed (HELMET only) |
| `ROPE_SCALING` | `false` | Enable RoPE scaling (LongPPL only) |
| `TASKS` | `crosslingual_longppl_en,crosslingual_longppl_fi` | lm-eval tasks (LongPPL only) |

## Container

Both scripts use the ROCm vLLM container at `/shared_silo/scratch/containers/`. The container needs `ulimit -n 65536` for HELMET's multiprocessing data loader — this is already set in the script.

Dependencies (`pytrec_eval`, `rouge_score`, etc.) are pip-installed into per-job temp dirs at runtime to avoid cross-job interference.
