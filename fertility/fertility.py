#!/usr/bin/env python3

import sys
import regex

from argparse import ArgumentParser

from transformers import AutoTokenizer


WORD_RE = regex.compile(r'[[:alnum:]]+|[^[:space:]]')


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--no-split', action='store_true',
                    help='process text without splitting lines')
    ap.add_argument('tokenizer')
    ap.add_argument('text')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    with open(args.text) as f:
        text = f.read()

    if args.no_split:
        lines = [text]
    else:
        lines = text.splitlines()

    token_count, word_count = 0, 0
    for line in lines:
        token_count += len(tokenizer(line).input_ids)
        word_count += len(WORD_RE.findall(line))

    print(f'fertility {token_count}/{word_count} ({token_count/word_count:.2f})')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
