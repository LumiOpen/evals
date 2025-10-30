#!/usr/bin/env python3
"""
Show HELMET results grouped by task and context length.
All individual dataset scores are shown under each task/context combination.
"""

import json
import os
import re
import sys
from collections import defaultdict

# Map dataset names to task names
DATASET_TO_TASK = {
    'ruler_niah_mk_2': 'recall',
    'ruler_niah_mk_3': 'recall',
    'ruler_niah_mv': 'recall',
    'json_kv': 'recall',
    'kilt_nq': 'rag',
    'kilt_triviaqa': 'rag',
    'kilt_hotpotqa': 'rag',
    'kilt_popqa': 'rag',
    'msmarco_rerank': 'rerank',
    'alce_asqa_': 'cite',
    'alce_qampari_': 'cite',
    'alce_asqa_nocite': 'alce_nocite',
    'alce_qampari_nocite': 'alce_nocite',
    'narrativeqa': 'longqa',
    'infbench_qa': 'longqa',
    'infbench_choice': 'longqa',
    'infbench_sum': 'summ',
    'icl_clinic': 'icl',
}

def get_task_name(dataset):
    for pattern, task in DATASET_TO_TASK.items():
        if dataset.startswith(pattern):
            return task
    return None

def extract_info(filename):
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
    if 'infbench_sum' in dataset:
        return 'rougeLsum_f1', score_data.get('rougeLsum_f1')

    for key in ['ruler_recall', 'exact_match', 'macro_f1', 'str_em', 'NDCG@10', 'accuracy']:
        if key in score_data:
            return key, score_data[key]

    return None, None

def main(results_dir):
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    score_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.score')])

    # Group by task and context
    grouped = defaultdict(lambda: defaultdict(list))

    for score_file in score_files:
        dataset, context = extract_info(score_file)
        if not dataset or not context:
            continue

        task = get_task_name(dataset)
        if not task:
            continue

        with open(os.path.join(results_dir, score_file)) as f:
            score_data = json.load(f)

        metric_name, metric_value = get_primary_metric(score_data, dataset)
        if metric_value is not None:
            grouped[task][context].append({
                'dataset': dataset,
                'metric': metric_name,
                'value': metric_value
            })

    # Print grouped results
    for task in ['recall', 'rag', 'rerank', 'cite', 'alce_nocite', 'longqa', 'summ', 'icl']:
        if task not in grouped:
            continue

        print(f"\n{'='*60}")
        print(f"TASK: {task.upper()}")
        print('='*60)

        for context in ['8k', '16k', '32k', '64k', '128k']:
            if context not in grouped[task]:
                continue

            scores = grouped[task][context]
            print(f"\n  {context}:")
            for s in scores:
                print(f"    {s['dataset']:<40} {s['metric']:<15} {s['value']:>6.2f}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python show_helmet_results.py <results_directory>")
        print("\nExample:")
        print("  python show_helmet_results.py output/v2/local/checkpoint_0001000")
        sys.exit(1)

    main(sys.argv[1])
