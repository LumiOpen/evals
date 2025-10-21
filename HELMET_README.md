# HELMET Integration

This document describes how to use HELMET (How to Evaluate Long-context Models Effectively and Thoroughly) evaluations in the evals framework.

## Overview

HELMET is a comprehensive benchmark for evaluating long-context language models across 7 task categories:
- **Recall**: Needle-in-a-haystack (NIAH), RULER benchmarks
- **RAG**: Retrieval-augmented generation tasks
- **Rerank**: Passage reranking tasks
- **Citation**: Citation generation and ALCE tasks
- **LongQA**: Long-form question answering
- **Summarization**: Document summarization tasks
- **ICL**: In-context learning tasks

Unlike other evaluations in this framework, HELMET runs as a standalone evaluation framework (not through lm-evaluation-harness), preserving its original execution model.

## Quick Start

### Running All HELMET Evaluations

To run all HELMET evaluations for a model:

```bash
sh helmet.sh /path/to/model
```

Or with a HuggingFace model ID:

```bash
sh helmet.sh meta-llama/Llama-3.1-8B-Instruct
```

### Using vLLM Backend

For faster inference with vLLM:

```bash
sh helmet.sh /path/to/model --backend vllm
```

### Quick Testing with Short Configs

For faster testing, use the `--short` flag to run shortened versions:

```bash
sh helmet.sh /path/to/model --short
```

Or use the quick test script:

```bash
sh helmet-quick.sh /path/to/model
```

### Running Individual HELMET Tasks

You can run specific HELMET tasks using `main.py`:

```bash
# Recall tasks
python main.py --model /path/to/model helmet_recall
python main.py --model /path/to/model helmet_niah
python main.py --model /path/to/model helmet_ruler

# RAG tasks
python main.py --model /path/to/model helmet_rag

# Reranking tasks
python main.py --model /path/to/model helmet_rerank

# Citation tasks
python main.py --model /path/to/model helmet_cite
python main.py --model /path/to/model helmet_alce_nocite

# Long QA tasks
python main.py --model /path/to/model helmet_longqa

# Summarization tasks
python main.py --model /path/to/model helmet_summ

# In-context learning tasks
python main.py --model /path/to/model helmet_icl
```

## Available HELMET Configurations

Each task category has multiple configuration variants:

### Recall Tasks
- `helmet_recall` - Full recall benchmark
- `helmet_recall_short` - Shortened version for quick testing
- `helmet_recall_vllm` - Optimized for vLLM backend
- `helmet_niah` - Needle-in-a-haystack tasks
- `helmet_niah_long` - Long-context NIAH tasks
- `helmet_ruler` - RULER benchmark
- `helmet_ruler_short` - Shortened RULER benchmark
- `helmet_recall_demo` - Demo configuration for testing

### RAG Tasks
- `helmet_rag` - Full RAG benchmark
- `helmet_rag_short` - Shortened version
- `helmet_rag_vllm` - Optimized for vLLM backend

### Reranking Tasks
- `helmet_rerank` - Full reranking benchmark
- `helmet_rerank_short` - Shortened version

### Citation Tasks
- `helmet_cite` - Citation generation tasks
- `helmet_cite_short` - Shortened version
- `helmet_alce_nocite` - ALCE tasks without citation
- `helmet_alce_nocite_short` - Shortened ALCE tasks

### Long QA Tasks
- `helmet_longqa` - Long-form QA benchmark
- `helmet_longqa_short` - Shortened version

### Summarization Tasks
- `helmet_summ` - Summarization benchmark
- `helmet_summ_short` - Shortened version

### In-Context Learning Tasks
- `helmet_icl` - ICL benchmark
- `helmet_icl_short` - Shortened version

## SLURM Configuration

HELMET evaluations use the same SLURM configuration options as other evaluations:

```bash
python main.py \
    --partition standard-g \
    --time 24:00:00 \
    --gres gpu:mi250:4 \
    --model /path/to/model \
    helmet_recall
```

## Backend Options

HELMET supports multiple backends:

### HuggingFace (default)
```bash
python main.py --model /path/to/model --backend hf helmet_recall
```

### vLLM
```bash
python main.py --model /path/to/model --backend vllm helmet_recall
```

### Custom Model Arguments
```bash
python main.py --model /path/to/model \
    --model_args "dtype=float16,max_model_len=131072" \
    helmet_recall
```

## Output

HELMET results are saved to the standard output directory structure:
```
output/v2/<model_org>/<model_name>/helmet_<task_name>.json
```

Each HELMET evaluation creates:
- A main JSON file with evaluation metadata
- Individual result files for each dataset in the HELMET output directory
- `.json.score` files with aggregated metrics

## Data Requirements

HELMET requires downloading benchmark datasets before first use. The HELMET framework will automatically download required datasets when running evaluations. Data is cached in `helmet/data/` directory.

To manually prepare datasets, you can follow the instructions in the [HELMET repository](https://github.com/princeton-nlp/HELMET).

## Troubleshooting

### Missing Dependencies

If you encounter missing dependencies, they will be automatically installed in the container during job execution. The required packages are:
- torch
- datasets
- transformers
- accelerate
- sentencepiece
- pytrec_eval
- rouge_score
- openai

### GPU Memory Issues

For very long contexts, you may need to adjust:
1. GPU allocation: `--gres gpu:mi250:8` (use more GPUs)
2. Memory utilization: `--model_args "gpu_memory_utilization=0.95"`
3. Context length: `--model_args "max_model_len=65536"` (reduce if needed)

### Job Timeout

Long-context evaluations can take significant time. Adjust the time limit as needed:
```bash
python main.py --time 48:00:00 --model /path/to/model helmet_recall
```

## Integration Details

HELMET is integrated as a standalone evaluation framework:
- Source code: `helmet/` directory (cloned from https://github.com/princeton-nlp/HELMET)
- Harness class: `HELMETHarness` in `evals/harnesses.py`
- SLURM template: `templates/helmet.sh`
- Eval configs: Registered in `evals/evals.py` with `HELMETConfig`

The integration preserves HELMET's original execution model while running it within the same singularity container environment used by other evaluations.

## References

- [HELMET Repository](https://github.com/princeton-nlp/HELMET)
- [HELMET Paper](https://arxiv.org/abs/2410.02694) (if available)

## Notes

- HELMET evaluations are designed for long-context models (typically 32k-128k+ context windows)
- Short configs use reduced test samples for faster iteration during development
- The vLLM backend is recommended for faster inference on long sequences
- Results may vary based on context length limitations of the model being evaluated
