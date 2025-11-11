#!/usr/bin/env python3
"""
Analyze the 3% discrepancy between our LongPPL results and the paper.

Our Results vs Paper:
- Qwen2-7B: LongPPL 2.23 vs 2.16 (+3.2%), PPL 4.96 vs 4.82 (+2.9%)
- Mistral-7B-Instruct-v0.2: LongPPL 2.08 vs 2.11 (-1.4%), PPL 4.27 vs 4.14 (+3.1%)

Key observation: Standard PPL is ~3% higher in BOTH models, suggesting systematic difference.
"""

import json
import sys
from pathlib import Path

# Our results
our_results = {
    "Qwen/Qwen2-7B": {
        "longppl": 2.23,
        "ppl": 4.96,
        "samples": 50
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "longppl": 2.08,
        "ppl": 4.27,
        "samples": 50
    }
}

# Paper results
paper_results = {
    "Qwen/Qwen2-7B": {
        "longppl": 2.16,
        "ppl": 4.82
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "longppl": 2.11,
        "ppl": 4.14
    }
}

print("=" * 80)
print("LongPPL DISCREPANCY ANALYSIS")
print("=" * 80)
print()

print("RESULTS COMPARISON:")
print("-" * 80)
for model in our_results:
    print(f"\n{model}:")
    print(f"  LongPPL: {our_results[model]['longppl']:.4f} vs {paper_results[model]['longppl']:.4f} (paper)")
    longppl_diff = (our_results[model]['longppl'] - paper_results[model]['longppl']) / paper_results[model]['longppl'] * 100
    print(f"           Difference: {longppl_diff:+.2f}%")

    print(f"  PPL:     {our_results[model]['ppl']:.4f} vs {paper_results[model]['ppl']:.4f} (paper)")
    ppl_diff = (our_results[model]['ppl'] - paper_results[model]['ppl']) / paper_results[model]['ppl'] * 100
    print(f"           Difference: {ppl_diff:+.2f}%")
    print(f"  Samples: {our_results[model]['samples']}")

print()
print("=" * 80)
print("PATTERN ANALYSIS:")
print("=" * 80)

# Check if PPL is systematically higher
ppl_diffs = []
for model in our_results:
    ppl_diff = (our_results[model]['ppl'] - paper_results[model]['ppl']) / paper_results[model]['ppl'] * 100
    ppl_diffs.append(ppl_diff)

avg_ppl_diff = sum(ppl_diffs) / len(ppl_diffs)
print(f"\nAverage PPL difference: {avg_ppl_diff:+.2f}%")
print(f"PPL is SYSTEMATICALLY ~{abs(avg_ppl_diff):.1f}% higher in both models")
print()
print("This suggests a systematic evaluation difference, NOT random variation!")

print()
print("=" * 80)
print("POSSIBLE ROOT CAUSES:")
print("=" * 80)
print("""
1. DATASET SAMPLING:
   - Paper may use different subset of GovReport
   - Different random seed for selecting 50 samples
   - Different filtering criteria (dataset-min-tokens)

2. MODEL CHECKPOINT VERSIONS:
   - Qwen2-7B may have multiple checkpoints
   - Mistral-7B-Instruct-v0.2 may have been updated
   - Different model precision (bf16 vs fp16)

3. TOKENIZATION DIFFERENCES:
   - Paper uses pre-tokenized dataset with Llama-2-7b tokenizer
   - We tokenize on-the-fly (maybe different?)
   - Check: emozilla/govreport-test-tokenized tokenizer

4. EVALUATION PARAMETERS:
   - Verify alpha=2.0, beta=-2.0 (matches paper)
   - Verify trunc_len=4096, sliding_window=1024
   - Check if paper uses different max_length

5. EVALUATOR MODEL:
   - We use meta-llama/Llama-3.1-8B (correct)
   - But which checkpoint? Base or instruct?
   - Paper doesn't specify exact checkpoint

6. BATCH SIZE / INFERENCE SETTINGS:
   - Different batch size in computation
   - Different attention implementation
   - Flash Attention 2 vs other implementations
""")

print()
print("=" * 80)
print("NEXT INVESTIGATION STEPS:")
print("=" * 80)
print("""
1. Check LongPPL repo for EXACT evaluation command used in paper
2. Verify the exact dataset: is emozilla/govreport-test-tokenized the right one?
3. Check if paper specifies number of samples (we assume 50)
4. Compare our log files to see if key token counts match expected
5. Check model card for Llama-3.1-8B - base vs instruct version
6. Try running with exact same random seed as paper (if available)
""")

print()
print("=" * 80)
print("CRITICAL QUESTION:")
print("=" * 80)
print("""
WHY is standard PPL ~3% higher in BOTH models but LongPPL is mixed?
- Qwen: +3.2% LongPPL, +2.9% PPL
- Mistral: -1.4% LongPPL, +3.1% PPL

Hypothesis: If key token identification is slightly different (due to evaluator
model or parameters), but overall perplexity computation is consistent, this
would explain why PPL is systematically higher while LongPPL varies.

This suggests the issue is in KEY TOKEN IDENTIFICATION, not perplexity computation!
""")

print()
