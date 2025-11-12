# RULER Long Context Evaluation

This document describes the integration of RULER (Rule-based Long-context Understanding Evaluation) benchmarks into the evaluation framework.

## What is RULER?

RULER is a benchmark designed to evaluate language models on their ability to handle long contexts. It includes **13 distinct subtasks** that test different aspects of long-context understanding:

### NIAH (Needle in a Haystack) - 8 variants
- **niah_single_1, niah_single_2, niah_single_3**: Finding specific information within long contexts
- **niah_multikey_1, niah_multikey_2, niah_multikey_3**: Finding multiple related pieces of information
- **niah_multivalue**: Retrieving multiple values for the same key
- **niah_multiquery**: Answering multiple queries about the same context

### Other Tasks - 5 types
- **ruler_vt**: Variable Tracking - Following variable assignments and retrieving values
- **ruler_cwe**: Common Words Extraction - Identifying frequently occurring words
- **ruler_fwe**: Frequent Words Extraction - Extracting the most frequent words
- **ruler_qa_hotpot**: Question Answering - Comprehension questions based on Hotpot dataset
- **ruler_qa_squad**: Question Answering - Comprehension questions based on SQuADv2 dataset

Each subtask is evaluated independently at each sequence length, allowing fine-grained analysis of model capabilities.

## Supported Sequence Lengths

The integration supports the following sequence lengths (powers of 2):

- 4,096 tokens
- 8,192 tokens
- 16,384 tokens
- 32,768 tokens (32K)
- 65,536 tokens (64K)
- 131,072 tokens (128K)

## Task Structure

The integration provides two modes:

1. **Granular Mode** (Recommended): Run individual subtasks at specific sequence lengths
   - Task format: `ruler_<subtask>_<seqlen>` (e.g., `ruler_niah_single_1_4096`)
   - 13 subtasks × 6 sequence lengths = **78 individual tasks**
   - Each task runs as a separate SLURM job
   - Allows parallel execution and fine-grained resource management

2. **Grouped Mode**: Run all subtasks at once for a sequence length
   - Task format: `ruler_<seqlen>` (e.g., `ruler_4096`)
   - All 13 subtasks run in a single job

## Usage

### Running All RULER Evaluations (78 jobs)

To run all subtasks at all sequence lengths (submits 78 SLURM jobs):

```bash
# Using HuggingFace backend
sh ruler.sh /path/to/model

# Using vLLM backend (faster, experimental)
sh ruler-vllm.sh /path/to/model
```

### Running Specific Subtasks

To run only certain subtasks across all or selected sequence lengths:

```bash
# Specific subtasks at all sequence lengths
sh ruler.sh /path/to/model --subtasks "niah_single_1,ruler_vt,ruler_qa_hotpot"

# Specific subtasks at specific sequence lengths
sh ruler.sh /path/to/model --subtasks "niah_single_1,ruler_vt" --sequence-lengths "4096,8192"

# With vLLM backend
sh ruler-vllm.sh /path/to/model --subtasks "niah_single_1,niah_single_2" --sequence-lengths "4096,8192,16384"
```

### Running Specific Sequence Lengths

To run all subtasks at only certain sequence lengths:

```bash
# All subtasks at specific sequence lengths
sh ruler.sh /path/to/model --sequence-lengths "4096,8192,16384"

# Submits 13 × 3 = 39 jobs
```

### Running Individual Tasks

To run specific individual task combinations:

```bash
# Single specific task
python main.py --model /path/to/model ruler_niah_single_1_4096

# Multiple specific tasks
python main.py --model /path/to/model \
    ruler_niah_single_1_4096 \
    ruler_ruler_vt_4096 \
    ruler_ruler_qa_hotpot_8192

# Grouped task (all subtasks at one sequence length)
python main.py --model /path/to/model ruler_4096
```

### Custom SLURM Configuration

You can customize the SLURM job parameters:

```bash
# Custom partition and time
sh ruler.sh /path/to/model --partition standard-g --time 24:00:00

# Custom GPU allocation
sh ruler.sh /path/to/model --gres gpu:mi250:8

# Combine multiple options
sh ruler-vllm.sh /path/to/model \
    --subtasks "niah_single_1,vt" \
    --sequence-lengths "4096,8192" \
    --partition standard-g \
    --time 12:00:00 \
    --gres gpu:mi250:8
```

**Important**: The scripts call `python main.py`, which internally generates SLURM job scripts and submits them using `sbatch`. Each call to `main.py` results in a separate SLURM job being queued. The scripts do NOT run the evaluations directly - they submit them to your HPC's SLURM scheduler.

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

### Example 1: Quick Test with Few Subtasks

Test a few subtasks with limited samples:

```bash
# Test 2 subtasks at 2 sequence lengths (4 jobs total)
sh ruler.sh /path/to/model \
    --subtasks "niah_single_1,vt" \
    --sequence-lengths "4096,8192" \
    --limit 10
```

### Example 2: Full NIAH Evaluation

Run all Needle in a Haystack variants at all sequence lengths:

```bash
# 8 NIAH subtasks × 6 sequence lengths = 48 jobs
sh ruler-vllm.sh /path/to/model \
    --subtasks "niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery"
```

### Example 3: Evaluate Short Context Lengths Only

Test all subtasks but only at shorter sequence lengths:

```bash
# 13 subtasks × 3 sequence lengths = 39 jobs
sh ruler.sh /path/to/model --sequence-lengths "4096,8192,16384"
```

### Example 4: Complete Workflow

```bash
# 1. Start by testing a small subset
sh ruler.sh /path/to/model \
    --subtasks "niah_single_1" \
    --sequence-lengths "4096" \
    --limit 10

# 2. Monitor to ensure it works
python watch.py --once

# 3. Once confirmed, run all evaluations (78 jobs)
sh ruler-vllm.sh /path/to/model

# 4. Monitor progress
python watch.py

# 5. View results when complete
sh summary.sh output/v2/<model-name>
```

### Example 5: Specific Individual Tasks

Run only specific task combinations manually:

```bash
# Run 3 specific tasks
python main.py --model /path/to/model \
    ruler_niah_single_1_4096 \
    ruler_ruler_vt_8192 \
    ruler_ruler_qa_hotpot_16384
```

### Example 6: Production Run with Optimal Settings

```bash
# Full evaluation with custom SLURM settings
sh ruler-vllm.sh org/modelname \
    --partition standard-g \
    --time 48:00:00 \
    --gres gpu:mi250:8
```

## References

- RULER Paper: [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)
- lm-evaluation-harness: [GitHub Repository](https://github.com/EleutherAI/lm-evaluation-harness)
- RULER Tasks: [GitHub Directory](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/ruler)

