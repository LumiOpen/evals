#!/usr/bin/env python3
"""
Extract HELMET results using OFFICIAL metrics with fallbacks to best available
Modified to accept directory as argument
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import re

# Official HELMET metrics with fallbacks
DATASET_METRICS = {
    # CITE tasks - str_em (per-dataset)
    'alce_asqa_': [('cite', 'asqa', ['str_em'])],
    'alce_qampari_': [('cite', 'qampari', ['str_em'])],

    # ALCE_NOCITE tasks - str_em
    'alce_asqa_nocite': [('alce_nocite', 'asqa', ['str_em'])],

    # ICL tasks - exact_match (per-dataset)
    'icl_trec_coarse_': [('icl', 'trec_coarse', ['exact_match'])],
    'icl_trec_fine_': [('icl', 'trec_fine', ['exact_match'])],
    'icl_banking77_': [('icl', 'banking77', ['exact_match'])],
    'icl_clinic': [('icl', 'clinic', ['exact_match'])],
    'icl_nlu_': [('icl', 'nlu', ['exact_match'])],

    # LONGQA tasks - prefer GPT-4, fallback to rougeL or f1
    'narrativeqa_': [('longqa', 'narrativeqa', ['gpt-4-score', 'f1', 'rougeL_f1'])],
    'infbench_qa_eng': [('longqa', 'infbench_qa', ['rougeL_f1', 'f1'])],
    'infbench_choice_eng': [('longqa', 'infbench_choice', ['exact_match', 'f1'])],

    # RAG tasks - substring_exact_match, fallback to f1
    'kilt_nq_': [('rag', 'nq', ['substring_exact_match', 'f1'])],
    'kilt_triviaqa_': [('rag', 'triviaqa', ['substring_exact_match', 'f1'])],
    'kilt_hotpotqa_': [('rag', 'hotpotqa', ['substring_exact_match', 'f1'])],
    'kilt_popqa': [('rag', 'popqa', ['substring_exact_match', 'f1'])],

    # RECALL tasks - ruler_recall or substring_exact_match
    'ruler_niah_mk_2': [('recall', 'niah_mk_2', ['ruler_recall'])],
    'ruler_niah_mk_3': [('recall', 'niah_mk_3', ['ruler_recall'])],
    'ruler_niah_mv': [('recall', 'niah_mv', ['ruler_recall'])],
    'json_kv_': [('recall', 'json_kv', ['substring_exact_match', 'exact_match'])],

    # RERANK tasks - NDCG@10
    'msmarco_rerank': [('rerank', 'msmarco', ['NDCG@10'])],

    # SUMM tasks - prefer gpt-4-f1, fallback to f1 or rougeLsum_f1 (per-dataset)
    'infbench_sum_eng': [('summ', 'infbench', ['gpt-4-f1', 'f1', 'rougeLsum_f1'])],
    'multi_lexsum_': [('summ', 'multi_lexsum', ['gpt-4-f1', 'f1', 'rougeLsum_f1'])],
}

def extract_context_length(filename):
    """Extract context length from filename."""
    match = re.search(r'_in(\d+)_', filename)
    if match:
        context_num = int(match.group(1))
        if context_num == 8192:
            return '8k'
        elif context_num == 16384:
            return '16k'
        elif context_num == 32768:
            return '32k'
        elif context_num == 65536:
            return '64k'
        elif context_num == 131072:
            return '128k'
    return None

def identify_task_and_metrics(filename):
    """Identify task category, subtask, and metric priority list from filename."""
    # Check alce_nocite first before alce (more specific pattern first)
    if 'alce_asqa_nocite' in filename:
        return 'alce_nocite', 'asqa', ['str_em']

    for pattern, configs in DATASET_METRICS.items():
        if pattern in filename:
            category, subtask, metrics = configs[0]
            return category, subtask, metrics
    return None, None, None

def get_best_available_metric(score_data, metric_list):
    """Get the first available metric from the priority list."""
    for metric in metric_list:
        if metric in score_data:
            return metric, round(score_data[metric], 2)
    return None, None

def main(results_dir):
    # Collect results: results[task_name][context] = (metric, value)
    results = defaultdict(lambda: defaultdict(dict))

    score_dir = Path(results_dir)
    if not score_dir.exists():
        print(f"Directory not found: {results_dir}\n")
        sys.exit(1)

    for score_file in score_dir.glob('*.score'):
        try:
            with open(score_file) as f:
                score_data = json.load(f)

            filename = score_file.name
            category, subtask, metric_list = identify_task_and_metrics(filename)
            context = extract_context_length(filename)

            if not all([category, subtask, metric_list, context]):
                continue

            # Get the best available metric
            metric_name, value = get_best_available_metric(score_data, metric_list)

            if metric_name is None or value is None:
                continue

            # Create task name - keep recall tasks separate
            if category == 'cite':
                task_name = f'cite-{subtask}'
            elif category == 'alce_nocite':
                task_name = f'alce_nocite-{subtask}'
            elif category == 'recall':
                task_name = f'recall-{subtask}'
            elif category in ['longqa', 'rag']:
                task_name = f'{category}-{subtask}'
            else:
                task_name = f'{category}-{subtask}'

            results[task_name][context] = {
                'metric': metric_name,
                'value': value
            }

        except Exception as e:
            print(f"Error processing {score_file.name}: {e}", file=sys.stderr)

    # Print results in tab-separated format
    contexts = ['8k', '16k', '32k', '64k', '128k']

    # Define task order matching HELMET structure (per-dataset split)
    task_order = [
        # CITE (per-dataset)
        'cite-asqa',
        'cite-qampari',
        # ALCE_NOCITE
        'alce_nocite-asqa',
        # ICL (per-dataset)
        'icl-trec_coarse',
        'icl-trec_fine',
        'icl-banking77',
        'icl-clinic',
        'icl-nlu',
        # LONGQA (per-dataset)
        'longqa-narrativeqa',
        'longqa-infbench_qa',
        'longqa-infbench_choice',
        # RAG (per-dataset)
        'rag-nq',
        'rag-triviaqa',
        'rag-hotpotqa',
        'rag-popqa',
        # RERANK
        'rerank-msmarco',
        # SUMM (per-dataset)
        'summ-infbench',
        'summ-multi_lexsum',
        # RECALL removed
    ]

    print("task\tmetric\t8k\t16k\t32k\t64k\t128k")

    for task in task_order:
        if task not in results:
            continue

        task_results = results[task]

        # Get the metric name (should be same across contexts)
        metric = None
        for ctx in contexts:
            if ctx in task_results:
                metric = task_results[ctx]['metric']
                break

        if not metric:
            continue

        row_values = [task, metric]
        for ctx in contexts:
            if ctx in task_results:
                row_values.append(str(task_results[ctx]['value']))
            else:
                row_values.append('-')

        print('\t'.join(row_values))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python helmet_extract_yarn.py <results_directory>")
        sys.exit(1)

    main(sys.argv[1])
