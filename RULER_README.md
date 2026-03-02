# RULER Long Context Evaluation

This document describes the integration of RULER (Rule-based Long-context Understanding Evaluation) benchmarks into the evaluation framework.

## Quick Reference

ruler.sh and ruler-vllm.sh are interchangeble in the command line. They just run different backends. 

```bash
# Run specific subtasks at specific lengths
sh ruler.sh /path/to/model --subtasks "niah_single_1,ruler_vt" --sequence-lengths "4096,8192"

# Run all NIAH tests at all lengths (48 jobs)
sh ruler.sh /path/to/model --subtasks "niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery"

# Run everything (78 jobs)
sh ruler-vllm.sh /path/to/model

# View results
sh summary_ruler.sh output/v2/<model-name>
```

**13 Subtasks:** `niah_single_1`, `niah_single_2`, `niah_single_3`, `niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3`, `niah_multivalue`, `niah_multiquery`, `ruler_vt`, `ruler_cwe`, `ruler_fwe`, `ruler_qa_hotpot`, `ruler_qa_squad`

**6 Sequence Lengths:** 4096, 8192, 16384, 32768, 65536, 131072

---

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
   - Task format: Run ruler subtask on given sequence length
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

**Important**: Scripts submit SLURM jobs via `python main.py` → `sbatch`. They don't run evaluations locally.

### Backend Selection

Choose between different backends based on your needs:

```bash
# HuggingFace backend (default, most compatible)
sh ruler.sh /path/to/model

# vLLM backend (faster inference)
sh ruler-vllm.sh /path/to/model

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
# RULER-specific summary (recommended for RULER results)
sh summary_ruler.sh output/v2/<model-name>

# General benchmark summary (for non-RULER benchmarks)
sh summary.sh output/v2/<model-name>

# List individual result files
ls output/v2/<model-name>/ruler_*.json
```

Result files are named: `ruler_<subtask>_<seqlen>.json` or `vllm_ruler_<subtask>_<seqlen>.json`

## Examples

### Example 1: Quick Test with Few Subtasks

Test a few subtasks with limited samples:

```bash
# Test 2 subtasks at 2 sequence lengths (4 jobs total)
sh ruler.sh /path/to/model \
    --subtasks "niah_single_1,ruler_vt" \
    --sequence-lengths "4096,8192" \
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

# 2. Monitor to ensure it works
python watch.py --once

# 3. Once confirmed, run all evaluations (78 jobs)
sh ruler-vllm.sh /path/to/model

# 4. Monitor progress
python watch.py

# 5. View results when complete
sh summary_ruler.sh output/v2/<model-name>
```

## References

- RULER Paper: [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)
- lm-evaluation-harness: [GitHub Repository](https://github.com/EleutherAI/lm-evaluation-harness)
- RULER Tasks: [GitHub Directory](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/ruler)

