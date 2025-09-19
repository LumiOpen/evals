# LUMI Evaluation Harness

A SLURM-based evaluation system for running language model benchmarks on the LUMI supercomputer. Supports both HuggingFace Transformers and vLLM backends for efficient model evaluation.

## Quick Start

```bash
# Basic evaluation with HuggingFace backend (default)
python main.py arc_challenge --model LumiOpen/Llama-Poro-2-70B-base

# Evaluation with vLLM backend for faster inference
python main.py arc_challenge --model LumiOpen/Llama-Poro-2-70B-base --backend vllm

# Multiple evaluations
python main.py arc_challenge hellaswag mmlu --model your-model-name
```

## Features

- **Multiple backends**: HuggingFace Transformers and vLLM support
- **SLURM integration**: Automatic job scheduling and resource management
- **Comprehensive benchmarks**: 100+ evaluation tasks including multilingual support
- **Result tracking**: Built-in monitoring and result aggregation tools
- **Container support**: Optimized Singularity containers for vLLM inference

## Usage

### Basic Command Structure

```bash
python main.py <eval_name> --model <model_path> [options]
```

### Backend Selection

- `--backend hf` (default): HuggingFace Transformers backend
- `--backend vllm`: vLLM backend with Singularity container
- `--backend auto`: Auto-selection (currently defaults to HF)

### Common Options

- `--model`: Path to model (local path or HuggingFace model ID)
- `--tokenizer`: Tokenizer path (defaults to model path)
- `--backend`: Inference backend (hf/vllm/auto)
- `--vllm_args`: Custom vLLM model arguments (only for vllm backend)
- `--partition`: SLURM partition (default: small-g)
- `--gres`: GPU resources (default: gpu:mi250:4)
- `--time`: Job time limit (default: 48:00:00)
- `--dryrun`: Generate script without submitting job

### Available Evaluations

**Core English benchmarks:**
- `arc_challenge`, `hellaswag`, `mmlu`, `truthfulqa_mc`, `winogrande`, `gsm8k`

**Multilingual benchmarks:**
- `arc_challenge_mt_fi`, `mmlu_mt_fi`, `gsm8k_mt_fi` (Finnish)
- `hellaswag_mt_sv`, `truthfulqa_mc_mt_da` (Nordic languages)
- And many more...

**Code benchmarks:**
- `humaneval_pass@1`, `humaneval_pass@10`
- `mbpp_pass@1`, `mbpp_pass@10`

**Finnish-specific:**
- `finbench_0shot`, `finbench_3shot`
- `include_finnish`, `ifeval_fi`

See `python main.py --help` for the complete list.

### Examples

```bash
# Standard HF evaluation
python main.py arc_challenge \
  --model LumiOpen/Llama-Poro-2-70B-base \
  --partition small-g \
  --time 24:00:00

# vLLM evaluation with custom GPU configuration
python main.py hellaswag \
  --model LumiOpen/Llama-Poro-2-70B-base \
  --backend vllm \
  --gres gpu:mi250:8 \
  --partition dev-g

# Multiple evaluations with chat template
python main.py arc_challenge hellaswag mmlu \
  --model your-chat-model \
  --apply_chat_template \
  --backend vllm

# Finnish language evaluations
python main.py finbench_3shot arc_challenge_mt_fi gsm8k_mt_fi \
  --model LumiOpen/Llama-Poro-2-70B-base

# Dry run to check generated script
python main.py arc_challenge \
  --model LumiOpen/Llama-Poro-2-70B-base \
  --backend vllm \
  --dryrun

# High-performance vLLM with custom settings
python main.py arc_challenge \
  --model LumiOpen/Llama-Poro-2-70B-base \
  --backend vllm \
  --vllm_args "max_model_len=8192,gpu_memory_utilization=0.95,max_num_batched_tokens=16384"
```

### Legacy Script Support

Common run configs can be found in command line scripts:

```bash
# Run common tests for Finnish CPT evals
sh cpt.sh /path/to/somemodel

# Other available scripts
sh all.sh /path/to/model      # Comprehensive evaluation suite
sh chat.sh /path/to/model     # Chat model evaluations
```

## Backend Details

### HuggingFace Backend

- Runs directly on LUMI compute nodes
- Uses `lm-evaluation-harness` with `--model hf`
- Supports all standard HF model formats
- Good for smaller models and development

### vLLM Backend

- Runs in optimized Singularity container
- Uses `lm-evaluation-harness` with `--model vllm`
- Includes ROCm optimizations for AMD GPUs
- Automatic tensor parallelism configuration
- Recommended for large models and production runs

Key vLLM features:
- Model prefetching and caching
- ROCm compiler optimizations
- Automatic GPU topology detection
- Memory-efficient inference

## Monitoring and Results

### Watch Running Jobs

```bash
# Monitor active jobs with live updates
python watch.py

# Check status once and exit
python watch.py --once

# Show recent job history
python watch.py --hist --days 7
```

Example output:
```bash
$ python watch.py --once
9678732 /path/to/model hellaswag_mt_fi is queued.
9678731 /path/to/model hellaswag is running.
Running loglikelihood requests:  43%|████▎     | 17337/40168 [13:41:50<16:37:51,  2.62s/it]
```

### View Results

```bash
# Summarize results for a model
./summary.sh ./output/v2/LumiOpen/Llama-Poro-2-70B-base/

# Generate reports
python report.py
```

Results are saved as JSON files in the output directory structure:
```
output/v2/{model_name}/{step_or_version}/{eval_name}.json
```

## File Structure

```
├── main.py                      # Main entry point
├── evals/
│   ├── evals.py                # Evaluation configurations
│   ├── harnesses.py            # Backend harness implementations
│   └── slurm.py               # SLURM integration utilities
├── templates/
│   ├── lm_eval_harness.sh      # HuggingFace template
│   ├── lm_eval_harness_vllm.sh # vLLM template
│   └── bigcode_eval_harness.sh # Code evaluation template
├── watch.py                    # Job monitoring tool
├── summary.sh                  # Results summary script
├── command_history.jsonl       # Job execution log
└── output/                     # Results directory
```

## Command History

All evaluations are logged in `command_history.jsonl`, which is used by monitoring scripts to track job status and report history.

Example entry:
```json
{
    "timestamp": "2023-11-09 08:21:12",
    "script_name": "/tmp/tmpvv66ri7g",
    "job_id": "4868114",
    "eval": "hellaswag",
    "model": "/path/to/model",
    "tokenizer": "/path/to/tokenizer",
    "backend": "vllm",
    "err_log": "/path/to/logs/4868114.err",
    "out_log": "/path/to/logs/4868114.out",
    "output_file": "/path/to/output/hellaswag.json"
}
```

Useful commands:
```bash
# Cancel all jobs for a specific model
grep /path/to/model command_history.jsonl | jq -r .job_id | xargs scancel

# Check results for recent jobs
python watch.py --hist --days 3
```

## Configuration

### SLURM Defaults

```bash
--project project_462000353
--partition small-g
--gres gpu:mi250:4
--time 48:00:00
```

### vLLM Container Configuration

The vLLM backend uses a pre-built container located at:
`/scratch/project_462000353/danizaut/containers/vllm_v10.1.1.sif.bak`

Model cache location: `/project/hf-cache/`

## Troubleshooting

### Common Issues

1. **"Model not found"**: Ensure model path is correct and accessible
2. **"SLURM job failed"**: Check logs in `./logs/` directory
3. **"Container not found"**: Verify vLLM container path in template
4. **"Permission denied"**: Check file permissions and SLURM account access

### Debugging

```bash
# Generate script without running
python main.py <eval> --model <model> --dryrun

# Check job logs
tail -f logs/latest.out
tail -f logs/latest.err

# View job history
python watch.py --hist
```

## Contributing

When adding new evaluations:

1. Add configuration to `evals/evals.py`
2. Ensure output format compatibility with existing tools
3. Test with both HF and vLLM backends if applicable
4. Update this README with new evaluation names

## License

This project is part of the LUMI-OpenGPU ecosystem.