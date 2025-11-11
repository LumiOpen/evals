# LongPPL Integration

This branch adds support for **LongPPL** (Long-context Perplexity) evaluation to the evaluation framework.

## What is LongPPL?

LongPPL is a novel perplexity metric from an ICLR 2025 paper: ["What is Wrong with Perplexity for Long-context Language Modeling?"](https://github.com/PKU-ML/LongPPL)

**Key insight:** Traditional perplexity shows near-zero correlation with actual long-context benchmark performance. LongPPL fixes this by focusing only on "key tokens" identified through long-short context comparison, achieving **-0.96 Pearson correlation** with long-context benchmarks.

**Why it matters:**
- **Fast screening**: Single perplexity run (30-60 min) can predict full HELMET performance (4-12 hours)
- **Cost effective**: ~72x cheaper than full evaluation suite
- **Reliable**: Strong correlation with actual long-context capabilities

## Quick Start

### Launch a single evaluation

```bash
# Standard 16k context evaluation (50 samples)
python main.py --backend hf --project project_462000963 \
  --partition standard-g --model meta-llama/Llama-3.1-8B \
  longppl_16k
```

### Launch multiple context lengths

```bash
# Evaluate at 8k, 16k, 32k
python main.py --backend hf --project project_462000963 \
  --partition standard-g --model meta-llama/Llama-3.1-8B \
  longppl_8k longppl_16k longppl_32k
```

### Quick iteration (25 samples)

```bash
# Faster evaluation with fewer samples
python main.py --backend hf --project project_462000963 \
  --partition standard-g --model meta-llama/Llama-3.1-8B \
  longppl_16k_quick
```

## Available Tasks

### Standard configurations (50 samples)
- `longppl_8k` - 8,192 token context
- `longppl_16k` - 16,384 token context
- `longppl_32k` - 32,768 token context
- `longppl_64k` - 65,536 token context
- `longppl_128k` - 131,072 token context

### Quick configurations (25 samples)
- `longppl_8k_quick`
- `longppl_16k_quick`
- `longppl_32k_quick`

### Full configurations (100 samples)
- `longppl_16k_full`
- `longppl_32k_full`

## Output Format

Results are saved to `output/v2/<model_org>/<model_name>/longppl_<context_length>.json`

Example output:
```json
{
  "model": "meta-llama/Llama-3.1-8B",
  "context_length": 16384,
  "dataset": "govreport-test-tokenized",
  "dataset_samples": 50,
  "alpha": 2.0,
  "beta": -2.0,
  "longppl": 12.34,
  "ppl": 45.67,
  "timestamp": "2025-01-11T09:30:00",
  "slurm_job_id": "14388020"
}
```

### Metrics Explained

- **longppl**: Perplexity computed only on "key tokens" (lower is better)
- **ppl**: Standard perplexity on all tokens (for comparison)
- **alpha**: Threshold for identifying key tokens (loss_short - loss_full > alpha)
- **beta**: Minimum loss threshold for key tokens (loss_full < -beta)

## Extracting Results

### Create TSV summary

```bash
# Basic TSV with longppl and ppl for all models
python longppl_extract_results.py

# Output: longppl_results.tsv
```

Example TSV output:
```tsv
model	longppl_8k	ppl_8k	longppl_16k	ppl_16k	longppl_32k	ppl_32k
meta-llama/Llama-3.1-8B	10.23	42.15	12.34	45.67	14.56	48.90
Qwen/Qwen2-7B	11.45	43.20	13.67	46.89	15.89	50.12
```

### Extract options

```bash
# List all found result files
python longppl_extract_results.py --list

# Custom output file
python longppl_extract_results.py --output my_results.tsv

# Exclude standard PPL (only LongPPL)
python longppl_extract_results.py --no-ppl

# Include sample counts
python longppl_extract_results.py --include-samples

# Custom base directory
python longppl_extract_results.py --base-dir /path/to/output
```

## Resource Requirements

### Compute resources (per model)

| Context | GPUs | Time | Memory |
|---------|------|------|--------|
| 8k-32k | 2 MI250 | 30-60 min | ~60GB |
| 64k | 2 MI250 | 60-90 min | ~80GB |
| 128k | 4 MI250 | 90-120 min | ~120GB |

**Comparison to HELMET:**
- LongPPL: 5 contexts × 1 hour = **5 GPU-hours**
- HELMET: 90 tasks × 4 hours = **360 GPU-hours**
- **Speedup: 72x cheaper**

### SLURM job configuration

The system automatically configures SLURM parameters:
- **Account**: From `--project` argument
- **Partition**: From `--partition` argument (typically `standard-g`)
- **GPUs**: Automatically determined (2 for most contexts, 4 for 128k)
- **Time**: 2 hours (sufficient for standard runs)

## Technical Details

### Dataset

LongPPL uses the **GovReport** dataset (tokenized):
- HuggingFace: `emozilla/govreport-test-tokenized`
- Contains long government reports (16k+ tokens)
- Standard benchmark for long-context evaluation

### Key Token Identification

The algorithm identifies "key tokens" by comparing model performance with full vs truncated context:

1. Compute loss with full long context: `loss_full`
2. Compute loss with truncated short context: `loss_short`
3. Mark token as "key" if:
   - `loss_short - loss_full > alpha` (token benefits from long context)
   - `loss_full < -beta` (token is well-predicted with full context)

Default thresholds:
- `alpha = 2.0`
- `beta = -2.0`

### Backend Support

**Supported:**
- ✅ HuggingFace backend (default, recommended)

**Not supported:**
- ❌ vLLM backend (LongPPL requires per-token loss computation)

If you specify `--backend vllm`, the system will automatically fall back to HuggingFace with a warning.

## Integration with HELMET

LongPPL complements HELMET evaluations:

1. **Checkpoint screening**: Run LongPPL on many checkpoints
2. **Select promising candidates**: High LongPPL correlation with HELMET
3. **Full evaluation**: Run complete HELMET suite on selected checkpoints

### Workflow example

```bash
# Step 1: Screen 20 checkpoints with LongPPL (100 GPU-hours)
for ckpt in checkpoint-{1000..20000..1000}; do
  python main.py --backend hf --model $ckpt longppl_16k
done

# Step 2: Analyze correlation, identify top 5 checkpoints
python longppl_extract_results.py
# (manual analysis of results)

# Step 3: Full HELMET evaluation on top 5 (1800 GPU-hours)
for ckpt in selected_checkpoints; do
  bash helmet_cpt.sh $ckpt
done

# Total: 1900 GPU-hours vs 7200 GPU-hours (full eval of all 20)
# Savings: 74%
```

## Validation

Before using LongPPL for production screening, validate the correlation holds for your model family:

```bash
# 1. Run LongPPL on 3-5 models you already have HELMET results for
python main.py --backend hf --model model1 longppl_16k
python main.py --backend hf --model model2 longppl_16k
python main.py --backend hf --model model3 longppl_16k

# 2. Extract results
python longppl_extract_results.py

# 3. Compare with HELMET aggregate scores
# (manual correlation analysis)

# 4. If correlation > 0.85, proceed with confidence
```

## Troubleshooting

### Job fails with "dataset not found"

The container automatically downloads GovReport on first run. If download fails:
- Check internet connectivity on compute node
- Manually pre-download: `huggingface-cli download emozilla/govreport-test-tokenized`

### "Could not parse longppl/ppl from output"

The evaluation completed but output parsing failed. Check log file:
```bash
cat output/v2/<model_org>/<model_name>/longppl_<context>.log
```

Look for lines like: `model_name: longppl: XX.XX, ppl: YY.YY`

### Results seem unrealistic (very high/low PPL)

- Check model is properly loaded (not using random weights)
- Verify context length doesn't exceed model's trained max length
- Try with fewer samples first (`longppl_16k_quick`)

### OOM (Out of Memory) errors

- Reduce context length (e.g., 128k → 64k)
- Request more GPUs (edit SLURM config or use larger `--partition`)
- Reduce batch size (requires modifying `longppl/perplexity/perplexity.py`)

## Implementation Details

### File structure

```
evals/
├── longppl/                        # LongPPL source code
│   ├── longppl.py                 # Core algorithm
│   ├── perplexity/                # Evaluation scripts
│   │   └── perplexity.py          # Main eval script
│   └── requirements.txt           # Dependencies
├── templates/
│   └── longppl.sh                 # SLURM template
├── evals/
│   ├── evals.py                   # Task definitions (LongPPLConfig)
│   └── harnesses.py               # Harness class (LongPPLHarness)
├── longppl_extract_results.py     # Result extraction script
└── README_longppl.md              # This file
```

### Dependencies

Added to container:
- `evaluate` - Evaluation framework
- `rouge_score` - ROUGE metric (transitive dependency)
- `pytrec_eval` - TREC evaluation metrics

All other dependencies (PyTorch, Transformers, datasets) already in vLLM container.

## References

- **Paper**: "What is Wrong with Perplexity for Long-context Language Modeling?" (ICLR 2025)
- **GitHub**: https://github.com/PKU-ML/LongPPL
- **Dataset**: https://huggingface.co/datasets/emozilla/govreport-test-tokenized

## Merge Status

**Branch**: `longppl-integration`

**Status**: Experimental - testing and validation in progress

**Merge criteria:**
- ✅ All tests pass
- ✅ Correlation with HELMET scores > 0.85
- ✅ No regression in existing HELMET tasks
- ✅ Documentation complete

To merge to main:
```bash
git checkout main
git merge longppl-integration
git push
```

## Contact

For questions or issues with LongPPL integration, check:
- LongPPL upstream: https://github.com/PKU-ML/LongPPL/issues
- This implementation: Check job logs in `/pfs/lustrep2/scratch/project_*/danizaut/evals/logs/`
