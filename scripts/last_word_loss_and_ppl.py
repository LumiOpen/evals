#!/usr/bin/env python3

import sys
import os
import json

import torch

from math import exp
from statistics import mean
from logging import warning
from argparse import ArgumentParser

from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM


def argparser():
    ap = ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('jsonl', nargs='+')
    return ap


def tokenize(tokenizer, text):
    return tokenizer(
        text,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_tensors='pt'
    )


class TooShortException(Exception):
    pass


def _metrics(text, prompt_length, tokenizer, model):
    loss_fct = CrossEntropyLoss(reduction='sum')

    encoded = tokenize(tokenizer, text).to(model.device)

    assert encoded.input_ids.shape[0] == 1
    if encoded.input_ids.shape[1] - prompt_length < 1:
        raise TooShortException()

    with torch.no_grad():
        output = model(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            labels=encoded.input_ids,
        )

    shift_logits = output.logits[:, prompt_length-1:-1, :]
    shift_labels = encoded.input_ids[:, prompt_length:]

    batch_size, seq_length, vocab_size = shift_logits.shape
    assert batch_size == 1

    total_loss = loss_fct(
        shift_logits.view(batch_size * seq_length, vocab_size),
        shift_labels.view(batch_size * seq_length)
    )

    char_length = len(tokenizer.decode(shift_labels[0]))
    loss = float(total_loss) / seq_length
    ppl = exp(loss)
    pplc = exp(float(total_loss)/char_length)
    
    return loss, ppl, pplc


def metrics(text, tokenizer, model):
    return _metrics(text, 1, tokenizer, model)


def conditional_metrics(prompt, output, tokenizer, model):
    encoded_prompt = tokenize(tokenizer, prompt)
    prompt_length = encoded_prompt.input_ids.shape[1]
    return _metrics(prompt+output, prompt_length, tokenizer, model)


def process(fn, tokenizer, model):
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            data = json.loads(l)
            text = data['text']
            prompt, output = text.rsplit(' ', 1)
            output = ' ' + output # not sure about this

            loss, ppl, pplc = conditional_metrics(prompt, output, tokenizer, model)
            print(f'{fn},{ppl:.2f},{loss:.2f},{pplc:.2f}')


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )

    for fn in args.jsonl:
        process(fn, tokenizer, model)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
