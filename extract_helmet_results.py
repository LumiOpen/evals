#!/usr/bin/env python3
"""
Extract all HELMET evaluation results from a results directory.
Reports individual dataset scores without any averaging or aggregation.

Usage:
    python extract_helmet_results.py <results_directory>
"""

import json
import os
import re
import sys


def extract_info(filename):
    """Extract dataset name and context length from filename."""
    match = re.search(r'_in(\d+)_', filename)
    if not match:
        return None, None

    context_len = int(match.group(1))
    context = f'{context_len//1024}k'

    match = re.match(r'^(.+?)_eval', filename)
    if match:
        dataset = match.group(1)
    else:
        dataset = filename.split('_eval')[0]

    return dataset, context


def get_primary_metric(score_data, dataset):
    """Get the primary metric for a dataset."""
    # Summ tasks should use rougeLsum_f1
    if 'infbench_sum' in dataset:
        return 'rougeLsum_f1', score_data.get('rougeLsum_f1')

    # Otherwise use standard priority
    for key in ['ruler_recall', 'exact_match', 'macro_f1', 'str_em', 'NDCG@10', 'accuracy']:
        if key in score_data:
            return key, score_data[key]

    return None, None


def extract_all_scores(results_dir):
    """Extract all individual scores without any aggregation."""
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    score_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.score')])

    if not score_files:
        print(f"Warning: No .score files found in {results_dir}", file=sys.stderr)
        return []

    results = []
    for score_file in score_files:
        dataset, context = extract_info(score_file)
        if not dataset or not context:
            continue

        with open(os.path.join(results_dir, score_file)) as f:
            score_data = json.load(f)

        metric_name, metric_value = get_primary_metric(score_data, dataset)
        if metric_value is not None:
            results.append({
                'dataset': dataset,
                'context': context,
                'metric': metric_name,
                'value': metric_value
            })

    return results


def print_table(results):
    """Print all results as a table."""
    print(f"{'Dataset':<50} {'Context':<8} {'Metric':<20} {'Score':<10}")
    print("=" * 90)

    for r in results:
        print(f"{r['dataset']:<50} {r['context']:<8} {r['metric']:<20} {r['value']:<10.2f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_helmet_results.py <results_directory>")
        print("\nExample:")
        print("  python extract_helmet_results.py output/v2/local/checkpoint_0001000")
        sys.exit(1)

    results_dir = sys.argv[1]
    results = extract_all_scores(results_dir)
    print_table(results)
