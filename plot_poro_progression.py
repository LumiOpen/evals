#!/usr/bin/env python3

import sys
import os
import re

import matplotlib.pyplot as plt

from argparse import ArgumentParser

from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import PercentFormatter


# regex for step from summary filename
STEP_RE = re.compile(r'.*step(\d+)')


# map model names into labels
LABEL_MAP = {
    'TurkuNLP_gpt3-finnish-8B': 'FinGPT 8B',
    'TurkuNLP_bloom-finnish-176b': 'BLUUMI (176B)',
}


# map labels into styles
STYLE_MAP = {
    'FinGPT 8B': { 'color': 'royalblue', 'linestyle': 'dashed' },
    'BLUUMI (176B)': { 'color': 'darkviolet', 'linestyle': 'dashdot' },
}


def argparser():
    ap = ArgumentParser()
    ap.add_argument(
        '--max-steps',
        type=int,
        default=238418,
    )
    ap.add_argument(
        '--output',
        default=None,
        metavar='FILE',
    )
    ap.add_argument(
        '--random-baseline',
        type=float,
        default=None,
    )
    ap.add_argument(
        'summaries',
        nargs='+',
        metavar='FILE',
    )
    return ap


def get_total(fn):
    with open(fn) as f:
        for l in f:
            if l.startswith('TOTAL'):
                return float(l.split()[1])
    raise Exception(f'missing TOTAL line in {fn}')


def get_label(fn):
    name = os.path.splitext(os.path.basename(fn))[0]
    for n, label in LABEL_MAP.items():
        if n in name:
            return label
    raise Exception(f'missing label for {fn}')


def percentage_formatter(x, pos):
    return f'{x:.0%}'


def main(argv):
    args = argparser().parse_args(argv[1:])

    # Expect progression for Poro, single results for others
    poro_results, other_results = {}, {}
    for fn in args.summaries:
        if 'poro-34b' in fn:
            m = STEP_RE.match(fn)
            step = int(m.group(1))
            trained = step / args.max_steps
            assert trained not in poro_results
            poro_results[trained] = get_total(fn)
        else:
            label = get_label(fn)
            assert label not in other_results
            other_results[label] = get_total(fn)

    if args.random_baseline is not None:
        poro_results[0] = args.random_baseline

    x, y = zip(*sorted(poro_results.items()))

    ax = plt.gca()
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    #ax.set_ylim([0.45, 0.65])    # TODO make arg
    
    plt.plot(x, y, marker='o', color='darkblue', linestyle='solid', label="Poro 34B")

    for label, y in other_results.items():
        style = STYLE_MAP[label]
        plt.axhline(y=y, label=label, **style)

    plt.xlabel('Poro 34B % trained')
    plt.ylabel('FIN-bench performance')

    plt.legend()

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
