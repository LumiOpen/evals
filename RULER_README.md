# RULER Long Context Evaluation

This document describes the integration of RULER (Rule-based Long-context Understanding Evaluation) benchmarks into the evaluation framework.

## What is RULER?

RULER is a benchmark designed to evaluate language models on their ability to handle long contexts. It tests models across various sequence lengths with different types of tasks:

- **NIAH (Needle in a Haystack)**: Finding specific information within long contexts
- **Variable Tracking**: Following variable assignments and retrieving values
- **Common Words Extraction**: Identifying frequently occurring words
- **Frequent Words Extraction**: Extracting the most frequent words
- **Question Answering**: Answering questions based on information in long contexts

## Supported Sequence Lengths

The integration supports the following sequence lengths (powers of 2):

- 4,096 tokens
- 8,192 tokens
- 16,384 tokens
- 32,768 tokens (32K)
- 65,536 tokens (64K)
- 131,072 tokens (128K)

## Usage

### Running All RULER Evaluations

To run all RULER evaluations across all sequence lengths:

```bash
# Using HuggingFace backend (default)
sh ruler.sh /path/to/model

# Using vLLM backend (faster, experimental)
sh ruler-vllm.sh /path/to/model
```

### Running Specific Sequence Lengths

To run evaluations for specific sequence lengths:

```bash
# Single sequence length
python main.py --model /path/to/model ruler_4096

# Multiple sequence lengths
python main.py --model /path/to/model ruler_4096 ruler_8192 ruler_16384
```

### Custom SLURM Configuration

You can customize the SLURM job parameters:

```bash
# Custom partition and time
sh ruler.sh /path/to/model --partition standard-g --time 24:00:00

# Custom GPU allocation
sh ruler.sh /path/to/model --gres gpu:mi250:8
```

### Backend Selection

Choose between different backends based on your needs:

```bash
# HuggingFace backend (default, most compatible)
sh ruler.sh /path/to/model

# vLLM backend (faster inference)
sh ruler-vllm.sh /path/to/model

# Custom model arguments for vLLM
python main.py --model /path/to/model --backend vllm \
    --model_args "max_model_len=32768,gpu_memory_utilization=0.95" \
    ruler_32768
```

## Resource Requirements

Different sequence lengths have different computational requirements:

| Sequence Length | Recommended Time | Recommended GPUs | Memory Requirements |
|-----------------|------------------|------------------|---------------------|
| 4,096           | 4-8 hours        | 4 GPUs           | ~40GB               |
| 8,192           | 8-12 hours       | 4 GPUs           | ~50GB               |
| 16,384          | 12-16 hours      | 4-8 GPUs         | ~70GB               |
| 32,768          | 16-24 hours      | 8 GPUs           | ~100GB              |
| 65,536          | 24-36 hours      | 8 GPUs           | ~150GB              |
| 131,072         | 36-48 hours      | 8 GPUs           | ~200GB              |

**Note**: These are estimates and may vary based on model size and architecture.

## Using Custom lm-evaluation-harness

If you need to use a specific version of lm-evaluation-harness that includes RULER tasks:

```bash
# Use a specific GitHub repository and branch
python main.py --model /path/to/model \
    --lm_eval https://github.com/EleutherAI/lm-evaluation-harness@main \
    ruler_4096

# Use a local development version
python main.py --model /path/to/model \
    --lm_eval /path/to/local/lm-evaluation-harness \
    ruler_4096
```

## Monitoring and Results

### Monitor Job Status

```bash
# Watch jobs in real-time
python watch.py

# Check status once
python watch.py --once

# View job history
python watch.py --hist --days 3
```

### View Results

```bash
# View summary of all results for a model
sh summary.sh output/v2/<model-name>

# Results are stored in:
# output/v2/<model-name>/<step>/ruler_<sequence_length>.json
```

## Interpreting Results

RULER evaluations produce accuracy scores for different task types. The overall score is an average across all subtasks. Higher scores indicate better long-context understanding:

- **0.0 - 0.3**: Poor long-context handling
- **0.3 - 0.5**: Basic long-context capabilities
- **0.5 - 0.7**: Good long-context performance
- **0.7 - 0.9**: Very good long-context performance
- **0.9 - 1.0**: Excellent long-context performance

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. Reduce batch size: `--batch_size 1`
2. Use vLLM backend with lower utilization: `--model_args "gpu_memory_utilization=0.85"`
3. Increase GPU allocation: `--gres gpu:mi250:8`
4. For very long sequences, consider gradient checkpointing if available

### RULER Tasks Not Found

If you see "Task not found" errors, ensure you're using a version of lm-evaluation-harness that includes RULER tasks:

```bash
python main.py --model /path/to/model \
    --lm_eval https://github.com/EleutherAI/lm-evaluation-harness@main \
    ruler_4096
```

### Very Long Run Times

For the longest sequence lengths (65K, 128K), consider:

1. Using vLLM backend for faster inference
2. Running in stages (test shorter lengths first)
3. Using `--limit` for testing: `--limit 50`

## Examples

### Complete Workflow

```bash
# 1. Run all RULER evaluations
sh ruler.sh /path/to/model

# 2. Monitor progress
python watch.py

# 3. View results when complete
sh summary.sh output/v2/<model-name>
```

### Quick Test

```bash
# Test with limited samples
python main.py --model /path/to/model --limit 10 ruler_4096
```

### Production Run

```bash
# Full evaluation with optimal settings
sh ruler-vllm.sh org/modelname \
    --partition standard-g \
    --time 48:00:00 \
    --gres gpu:mi250:8
```

## References

- RULER Paper: [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)
- lm-evaluation-harness: [GitHub Repository](https://github.com/EleutherAI/lm-evaluation-harness)
- RULER Tasks: [GitHub Directory](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/ruler)

