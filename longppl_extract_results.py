#!/usr/bin/env python3
"""
Extract LongPPL results from JSON files and create a TSV summary.

Usage:
    python longppl_extract_results.py [--output results.tsv]
"""

import argparse
import json
import os
import glob
from pathlib import Path
from collections import defaultdict

def find_longppl_results(base_dir="output/v2"):
    """Find all LongPPL result JSON files."""
    pattern = os.path.join(base_dir, "**", "longppl_*.json")
    return glob.glob(pattern, recursive=True)

def parse_longppl_file(filepath):
    """Parse a single LongPPL JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Extract model name from path or from JSON
        if 'model' in data:
            model = data['model']
        else:
            # Try to extract from path: output/v2/org/model/longppl_*.json
            parts = Path(filepath).parts
            if 'v2' in parts:
                v2_idx = parts.index('v2')
                if len(parts) > v2_idx + 2:
                    model = f"{parts[v2_idx + 1]}/{parts[v2_idx + 2]}"
                else:
                    model = parts[v2_idx + 1] if len(parts) > v2_idx + 1 else "unknown"
            else:
                model = "unknown"

        return {
            'model': model,
            'context_length': data.get('context_length'),
            'longppl': data.get('longppl'),
            'ppl': data.get('ppl'),
            'dataset_samples': data.get('dataset_samples'),
            'timestamp': data.get('timestamp'),
        }
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def extract_all_results(base_dir="output/v2"):
    """Extract all LongPPL results and organize by model."""
    files = find_longppl_results(base_dir)
    print(f"Found {len(files)} LongPPL result files")

    # Organize results by model
    results_by_model = defaultdict(dict)

    for filepath in files:
        result = parse_longppl_file(filepath)
        if result:
            model = result['model']
            ctx_len = result['context_length']

            # Store both longppl and ppl
            if ctx_len:
                results_by_model[model][f"longppl_{ctx_len}"] = result['longppl']
                results_by_model[model][f"ppl_{ctx_len}"] = result['ppl']
                results_by_model[model][f"samples_{ctx_len}"] = result['dataset_samples']

    return results_by_model

def format_context_name(ctx_len):
    """Convert context length to readable name (e.g., 8192 -> 8k)."""
    if ctx_len >= 1024:
        return f"{ctx_len // 1024}k"
    return str(ctx_len)

def create_tsv(results_by_model, output_file="longppl_results.tsv", include_ppl=True, include_samples=False):
    """Create a TSV file from the results."""

    # Determine all context lengths present
    all_contexts = set()
    for model_results in results_by_model.values():
        for key in model_results.keys():
            if key.startswith('longppl_'):
                ctx_len = int(key.split('_')[1])
                all_contexts.add(ctx_len)

    all_contexts = sorted(all_contexts)

    # Build header
    header = ["model"]
    for ctx_len in all_contexts:
        ctx_name = format_context_name(ctx_len)
        header.append(f"longppl_{ctx_name}")
        if include_ppl:
            header.append(f"ppl_{ctx_name}")
        if include_samples:
            header.append(f"samples_{ctx_name}")

    # Write TSV
    with open(output_file, 'w') as f:
        # Write header
        f.write('\t'.join(header) + '\n')

        # Write data rows
        for model in sorted(results_by_model.keys()):
            row = [model]
            model_results = results_by_model[model]

            for ctx_len in all_contexts:
                # LongPPL value
                longppl_key = f"longppl_{ctx_len}"
                longppl_val = model_results.get(longppl_key)
                row.append(f"{longppl_val:.4f}" if longppl_val is not None else "")

                if include_ppl:
                    # Standard PPL value
                    ppl_key = f"ppl_{ctx_len}"
                    ppl_val = model_results.get(ppl_key)
                    row.append(f"{ppl_val:.4f}" if ppl_val is not None else "")

                if include_samples:
                    # Sample count
                    samples_key = f"samples_{ctx_len}"
                    samples_val = model_results.get(samples_key)
                    row.append(str(samples_val) if samples_val is not None else "")

            f.write('\t'.join(row) + '\n')

    print(f"Results written to {output_file}")
    print(f"Models: {len(results_by_model)}")
    print(f"Context lengths: {', '.join(format_context_name(c) for c in all_contexts)}")

def main():
    parser = argparse.ArgumentParser(description="Extract LongPPL results to TSV")
    parser.add_argument('--output', '-o', default='longppl_results.tsv',
                        help='Output TSV file (default: longppl_results.tsv)')
    parser.add_argument('--base-dir', '-d', default='output/v2',
                        help='Base directory to search for results (default: output/v2)')
    parser.add_argument('--no-ppl', action='store_true',
                        help='Exclude standard PPL values from output')
    parser.add_argument('--include-samples', action='store_true',
                        help='Include sample counts in output')
    parser.add_argument('--list', action='store_true',
                        help='List all found result files and exit')

    args = parser.parse_args()

    if args.list:
        files = find_longppl_results(args.base_dir)
        print(f"Found {len(files)} LongPPL result files:")
        for f in sorted(files):
            print(f"  {f}")
        return

    # Extract and format results
    results = extract_all_results(args.base_dir)

    if not results:
        print("No LongPPL results found!")
        return

    # Create TSV
    create_tsv(results, args.output,
               include_ppl=not args.no_ppl,
               include_samples=args.include_samples)

if __name__ == '__main__':
    main()
