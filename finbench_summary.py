#!/usr/bin/env python3

import sys
import json

from copy import deepcopy
from statistics import mean
from numbers import Number
from logging import warning, error
from argparse import ArgumentParser


# FIN-bench task hierarchy, following https://github.com/TurkuNLP/FIN-bench/
# benchmark_tasks. Value is None for tasks without subtasks.
TASKS = {
    'analogies': None,
    'arithmetic' : {
        '1_digit_addition': None,
        '1_digit_division': None,
        '1_digit_multiplication': None,
        '1_digit_subtraction': None,
        '2_digit_addition': None,
        '2_digit_division': None,
        '2_digit_multiplication': None,
        '2_digit_subtraction': None,
        '3_digit_addition': None,
        '3_digit_division': None,
        '3_digit_multiplication': None,
        '3_digit_subtraction': None,
        '4_digit_addition': None,
        '4_digit_division': None,
        '4_digit_multiplication': None,
        '4_digit_subtraction': None,
        '5_digit_addition': None,
        '5_digit_division': None,
        '5_digit_multiplication': None,
        '5_digit_subtraction': None,
    },
    'cause_and_effect': {
        'one_sentence': None,
        'one_sentence_no_prompt': None,
        'two_sentences': None,
    },
    'emotions': None,
    'empirical_judgments': None,
    'general_knowledge': None,
    'hhh_alignment' : {
        'harmless': None,
        'helpful': None,
        'honest': None,
        'other': None,
    },
    'intent_recognition': None,
    'misconceptions': None,
    'paraphrase': None,
    'sentence_ambiguity': None,
    'similarities_abstraction': None,
}


def argparser():
    ap = ArgumentParser(description='summarize FIN-bench evaluation results')
    ap.add_argument(
        'json',
        help='results',
    )
    ap.add_argument(
        '--result-key',
        default='multiple_choice_grade',
        help='key for task results in json data'
    )
    ap.add_argument(
        '--prefix',
        default='bigbench_',
        help='task prefix in json data',
    )
    ap.add_argument(
        '--include-hhh',
        action='store_true',
        help='include HHH task results',
    )
    return ap


def organize_results(flat, tree, args):
    # Copy results from flat (dcit) to tree (dict), deleting them from flat
    for k, v in tree.items():
        if v is None:
            # Copy over result and delete original
            fk = f'{args.prefix}{k}'
            if fk not in flat:
                raise Exception(
                    f'missing "{fk}" in "results" in {args.json}'
                )
            if args.result_key not in flat[fk]:
                raise Exception(
                    f'missing "{args.result_key}" for "{fk}" in {args.json}'
                )
            tree[k] = flat[fk][args.result_key]
            del flat[fk]
        else:
            assert isinstance(v, dict)
            organize_results(flat, v, args)


def get_values(node, vals=None):
    if vals is None:
        vals = []
    if isinstance(node, Number):
        vals.append(node)
    else:
        assert isinstance(node, dict)
        for n in node.values():
            get_values(n, vals)
    return vals


def main(argv):
    args = argparser().parse_args(argv[1:])

    with open(args.json) as f:
        data = json.load(f)

    results = deepcopy(TASKS)

    try:
        organize_results(data['results'], results, args)
    except Exception as e:
        print(f'error: {e}')
        return

    if data['results']:
        print(f'error: extra results: {",".join(data["results"].keys())}')
        return

    if not args.include_hhh:
        warning('excluding HHH task results')
        del results['hhh_alignment']
    
    task_avgs = []
    for k, v in results.items():
        values = get_values(v)
        avg = mean(values)
        print(f'{k}\t{avg:.4f}') #\t{values}')
        task_avgs.append(avg)
    print(f'TOTAL\t{mean(task_avgs):.4f}')
    

if __name__ == '__main__':
    sys.exit(main(sys.argv))
