# 3% Discrepancy Root Cause Analysis

## Summary

Found the ROOT CAUSE of the 3% systematic discrepancy between our results and the paper!

## The Critical Difference

### Paper's Evaluation:
```bash
python perplexity.py \
    --dataset emozilla/govreport-test-tokenized \
    --tokenized \
    --dataset-min-tokens 16384 \
    --samples 50 \
    --model Qwen/Qwen2-7B \
    --evaluator-model meta-llama/Llama-3.1-8B \
    --mode online \
    --alpha 2.0 \
    --beta -2.0
    # NOTE: NO --max-length specified!
    # Uses default: max_length=32768
```

### Our Evaluation:
```bash
python perplexity.py \
    --dataset emozilla/govreport-test-tokenized \
    --tokenized \
    --dataset-min-tokens 16384 \
    --samples 50 \
    --model "$MODEL_LOCAL" \
    --evaluator-model meta-llama/Llama-3.1-8B \
    --mode online \
    --max-length 16384 \  # <-- WE SET THIS!
    --trunc-len 4096 \
    --sliding-window 1024 \
    --alpha 2.0 \
    --beta -2.0
```

## The Issue

In `longppl/perplexity/perplexity.py` line 23:
```python
encoded_texts = [x[0:args.max_length-1] for x in encodings["input_ids"]]
```

This **TRUNCATES** the input sequences to `max_length-1` tokens!

- **Paper**: Evaluates up to **32767 tokens** per sample
- **Us**: Evaluate only **16383 tokens** per sample

## Impact

We're evaluating on **HALF** the sequence length that the paper uses!

This explains:
1. **Systematic 3% higher PPL**: Shorter contexts often lead to higher perplexity because there's less context for prediction
2. **Mixed LongPPL results**: Key token identification is affected differently depending on where key tokens appear in the longer sequences

## Results Comparison

| Model | Metric | Ours (16k) | Paper (32k) | Diff |
|-------|--------|------------|-------------|------|
| Qwen2-7B | LongPPL | 2.23 | 2.16 | +3.2% |
| Qwen2-7B | PPL | 4.96 | 4.82 | +2.9% |
| Mistral-7B-v0.2 | LongPPL | 2.08 | 2.11 | -1.4% |
| Mistral-7B-v0.2 | PPL | 4.27 | 4.14 | +3.1% |

**Pattern**: PPL is systematically ~3% higher when evaluating on shorter sequences (16k vs 32k).

## Why This Happened

We conflated TWO different parameters:
1. **`--dataset-min-tokens`**: Filters which samples to use (keep only samples with ≥16384 tokens)
2. **`--max-length`**: Truncates samples during evaluation to this length

The paper uses:
- `--dataset-min-tokens 16384` (filter: only use long documents)
- `--max-length 32768` (default, evaluate up to 32k tokens)

We mistakenly set:
- `--dataset-min-tokens 16384` ✓
- `--max-length 16384` ✗ (should use default 32768!)

## The Confusion

Our task is named `longppl_16k` because we filter for documents with ≥16384 tokens (`--dataset-min-tokens`).

But this does NOT mean we should truncate evaluation to 16384 tokens (`--max-length`)!

The paper evaluates **much longer** than the minimum filter threshold.

## Solution

Remove `--max-length` from our template, or set it to 32768 to match paper's default.

Then we'll be evaluating on the same sequence lengths as the paper.

## Expected Outcome

After fixing this:
- PPL should drop by ~3% to match paper
- LongPPL should align more closely with paper values
- Discrepancy should reduce to <1% (within random variation)

## Additional Findings

Other parameters DO match the paper:
- ✓ evaluator_model: meta-llama/Llama-3.1-8B
- ✓ alpha: 2.0
- ✓ beta: -2.0
- ✓ trunc_len: 4096 (default)
- ✓ sliding_window: 1024 (default)
- ✓ samples: 50
- ✓ dataset_min_tokens: 16384
- ✓ mode: online

The ONLY difference is max_length: 16384 (us) vs 32768 (paper default).

## Conclusion

The 3% discrepancy is NOT a "minor thing" - it's a systematic evaluation difference caused by truncating sequences to 16k instead of 32k tokens.

This is a CONFIGURATION BUG in our implementation, not an acceptable validation tolerance.

Fix: Use `--max-length 32768` (or omit to use default) to match paper's evaluation setup.
