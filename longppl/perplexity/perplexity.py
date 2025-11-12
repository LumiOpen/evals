import argparse
import datasets
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from longppl.longppl import *
import os

# Compute LongPPL of input texts
def compute_perplexity(
    encodings, model, evaluator_model, tokenizer, evaluator_tokenizer, args, device=None
):
    if device is not None:
        assert device in ["gpu", "cpu",
                          "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoded_texts = [x[0:args.max_length-1] for x in encodings["input_ids"]]

    pbar = tqdm(total=len(encoded_texts))
    longppls, ppls, nums_key_token, nums_token = [], [], [], []

    def convert_tokenized_to_text(tokenized_input, tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        text = tokenizer.batch_decode(tokenized_input)
        return text

    for encoding_index in range(0, len(encoded_texts)):
        tokenized_input = torch.tensor(encoded_texts[encoding_index:encoding_index+1]).to(device)
        if args.tokenized:
            text = convert_tokenized_to_text(tokenized_input, args.tokenizer_path)
        else:
            text = convert_tokenized_to_text(tokenized_input, args.model)

        if not os.path.exists(os.path.join("key_text", args.evaluator_name)):
            os.makedirs(os.path.join("key_text", args.evaluator_name))
        save_path = os.path.join("key_text", args.evaluator_name, f"slice_{encoding_index}.txt")

        with torch.no_grad():
            output = compute_longppl(
                text=text[0], 
                model=model,
                evaluator_model=evaluator_model,
                tokenizer=tokenizer, 
                evaluator_tokenizer=evaluator_tokenizer, 
                save_path=save_path, 
                trunc_len=args.trunc_len, 
                sliding_window=args.sliding_window,
                alpha=args.alpha,
                beta=args.beta
            )
        longppl = output['longppl']
        ppl = output['ppl']
        n_key_token = output['n_key_token'] 
        n_token = output['n_token']
        
        if longppl is not None:
            longppls.append(longppl)
            nums_key_token.append(n_key_token)
        ppls.append(ppl)
        nums_token.append(n_token)
        longppl = np.exp((np.log(np.stack(longppls)) * np.stack(nums_key_token)).sum() / np.stack(nums_key_token).sum())
        ppl = np.exp((np.log(np.stack(ppls)) * np.stack(nums_token)).sum() / np.stack(nums_token).sum())

        pbar.set_postfix(longppl=longppl, ppl=ppl)
        pbar.update(1)

    return {"longppl": longppl, "ppl": ppl}


def main(args):
    print("=" * 80, flush=True)
    print("DEBUG: Starting main() function", flush=True)
    print(f"DEBUG: Model path: {args.model}", flush=True)
    print(f"DEBUG: Evaluator model: {args.evaluator_model}", flush=True)
    print(f"DEBUG: Mode: {args.mode}", flush=True)
    print(f"DEBUG: Dataset: {args.dataset}", flush=True)
    print("=" * 80, flush=True)

    print("\nDEBUG: [1/8] Loading target model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    print("DEBUG: [1/8] ✓ Target model loaded successfully", flush=True)

    print("DEBUG: [2/8] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("DEBUG: [2/8] ✓ Tokenizer loaded successfully", flush=True)

    if args.mode == 'online':
        print("DEBUG: [3/8] Loading evaluator model (online mode)...", flush=True)
        evaluator_model = AutoModelForCausalLM.from_pretrained(args.evaluator_model, torch_dtype=torch.bfloat16, device_map="auto")
        print("DEBUG: [3/8] ✓ Evaluator model loaded", flush=True)
        print("DEBUG: [4/8] Loading evaluator tokenizer...", flush=True)
        evaluator_tokenizer = AutoTokenizer.from_pretrained(args.evaluator_model)
        print("DEBUG: [4/8] ✓ Evaluator tokenizer loaded", flush=True)
    elif args.mode == 'offline':
        print("DEBUG: [3/8] Skipping evaluator model (offline mode)", flush=True)
        evaluator_model, evaluator_tokenizer = None, None

    print("DEBUG: [5/8] Loading dataset...", flush=True)
    if args.tokenized:
        print("DEBUG: Dataset is pre-tokenized", flush=True)
        try:
            input_texts = datasets.load_from_disk(args.dataset)
            print(f"DEBUG: Loaded from disk: {len(input_texts)} samples", flush=True)
        except:
            input_texts = datasets.load_dataset(
                args.dataset, name=args.subset, split=args.split)
            print(f"DEBUG: Loaded from HF: {len(input_texts)} samples", flush=True)
    else:
        # Check if dataset is a local HF Dataset directory
        if os.path.isdir(args.dataset) and os.path.exists(os.path.join(args.dataset, 'dataset_info.json')):
            print(f"DEBUG: Detected HF Dataset directory: {args.dataset}", flush=True)
            input_texts = datasets.load_from_disk(args.dataset)
            print(f"DEBUG: ✓ HF Dataset loaded from disk: {len(input_texts)} documents", flush=True)
        # Check if dataset is a local JSON file
        elif args.dataset.endswith('.json') and os.path.exists(args.dataset):
            print(f"DEBUG: Detected local JSON file: {args.dataset}", flush=True)
            print("DEBUG: Calling datasets.load_dataset('json', ...)...", flush=True)
            input_texts = datasets.load_dataset('json', data_files=args.dataset, field='documents', split='train')
            print(f"DEBUG: ✓ JSON loaded: {len(input_texts)} documents", flush=True)
        else:
            print(f"DEBUG: Loading HF dataset: {args.dataset}", flush=True)
            input_texts = datasets.load_dataset(
                args.dataset, name=args.subset, split=args.split)
            print(f"DEBUG: ✓ HF dataset loaded: {len(input_texts)} documents", flush=True)

        # Progress tracking for tokenization
        tokenize_counter = {'count': 0, 'total': len(input_texts)}

        def tokenize(example, idx):
            # Extract text from example dict if it's a dict, otherwise use example directly
            text_to_tokenize = example['text'] if isinstance(example, dict) else example
            tokenized = tokenizer(
                text_to_tokenize,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                max_length=sys.maxsize,
                return_attention_mask=True,
                return_offsets_mapping=True
            )
            example["input_ids"] = tokenized["input_ids"]
            example["attention_mask"] = tokenized["attention_mask"]
            example["tokenized_len"] = len(tokenized["input_ids"])
            example["offsets_mapping"] = tokenized["offsets_mapping"]

            # Print progress every 10 documents or at the end
            tokenize_counter['count'] = idx + 1
            if (idx + 1) % 10 == 0 or (idx + 1) == tokenize_counter['total']:
                print(f"Tokenized {idx + 1}/{tokenize_counter['total']} documents", flush=True)

            return example

        print("DEBUG: [6/8] Starting tokenization...", flush=True)
        print(f"DEBUG: Tokenizing {len(input_texts)} documents...", flush=True)
        input_texts = input_texts.map(tokenize, with_indices=True, desc="Tokenizing documents")
        print(f"DEBUG: [6/8] ✓ Tokenization complete: {len(input_texts)} documents", flush=True)

        if args.save_tokenized:
            print(f"DEBUG: Saving tokenized dataset to {args.save_tokenized}", flush=True)
            input_texts.save_to_disk(args.save_tokenized)
            print(f"DEBUG: ✓ Saved tokenized dataset", flush=True)
            return

    print("DEBUG: [5/8] ✓ Dataset loaded successfully", flush=True)
    print(f"DEBUG: Dataset size: {len(input_texts)} documents", flush=True)

    if args.dataset_min_tokens:
        print(f"DEBUG: [7/8] Filtering by min tokens: {args.dataset_min_tokens}...", flush=True)
        before_filter = len(input_texts)
        input_texts = input_texts.filter(
            lambda x: x["tokenized_len"] >= args.dataset_min_tokens)
        print(f"DEBUG: [7/8] ✓ Filtered: {before_filter} → {len(input_texts)} documents", flush=True)

    if args.samples:
        print(f"DEBUG: Limiting to first {args.samples} samples...", flush=True)
        input_texts = input_texts[:args.samples]
        print(f"DEBUG: ✓ Limited to {len(input_texts)} samples", flush=True)

    print(f"DEBUG: [8/8] Starting compute_perplexity with {len(input_texts)} documents...", flush=True)
    ppl = compute_perplexity(
        model=model,
        evaluator_model=evaluator_model,
        tokenizer=tokenizer,
        evaluator_tokenizer=evaluator_tokenizer,
        encodings=input_texts,
        args=args,
    )
    print(f"DEBUG: [8/8] ✓ Perplexity computation complete", flush=True)
    print("=" * 80, flush=True)
    print(f"FINAL RESULTS: {args.model}", flush=True)
    print(f"  LongPPL: {ppl['longppl']}", flush=True)
    print(f"  PPL: {ppl['ppl']}", flush=True)
    print("=" * 80, flush=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str,
        help="Repository name or local path to the model being evaluated"
    )

    parser.add_argument(
        "--evaluator-model", type=str,
        help="Repository name or local path to the evaluator model"
    )

    parser.add_argument(
        "--mode", type=str,
        choices=['online', 'offline'],
        default='offline',
        help="Set to 'offline' to use precomputed key tokens. Set to 'online' to compute key tokens using a custom LLM"
    )

    parser.add_argument(
        "--evaluator-name", type=str,
        default="Meta-Llama-3.1-8B",
        help="If mode is 'online', key tokens will be saved to perplexity/key_text/evaluator-name. "
             "If mode is 'offline', specify the name of a local folder containing precomputed key tokens. "
             "Default options include: Qwen2-72B-Instruct, Mistral-Large-Instruct-2407, Meta-Llama-3.1-8B"
    )

    parser.add_argument(
        "--dataset", type=str,
        help="Name or local path of the Hugging Face dataset"
    )

    parser.add_argument(
        "--tokenized", action='store_true',
        help="Set this flag if the dataset is already tokenized"
    )

    parser.add_argument(
        "--tokenizer-path", type=str,
        default="NousResearch/Llama-2-7b-hf",
        help="Path to the tokenizer used for processing the dataset (only used if --tokenized is set)"
    )

    parser.add_argument(
        "--subset", type=str,
        help="Subset name of the dataset (if applicable)"
    )

    parser.add_argument(
        "--max-length", type=int,
        default=32768,
        help="Maximum token length. Samples exceeding this will be truncated from the end"
    )

    parser.add_argument(
        "--dataset-min-tokens", type=int,
        help="If specified, removes all samples with fewer than this number of tokens"
    )

    parser.add_argument(
        "--split", type=str,
        default="test",
        help="Dataset split to use (e.g., 'train', 'validation', 'test')"
    )

    parser.add_argument(
        "--samples", type=int,
        help="If specified, only the first N samples from the dataset will be used"
    )

    parser.add_argument(
        "--save-tokenized", type=str,
        help="If specified, saves the tokenized dataset to the given path"
    )

    parser.add_argument(
        "--trunc-len", type=int,
        default=4096,
        help="Length of the short context window used in LongPPL calculation"
    )

    parser.add_argument(
        "--sliding-window", type=int,
        default=1024,
        help="Size of the sliding window used in LongPPL calculation"
             "(see Appendix A.1 in the paper for details)"
    )

    parser.add_argument(
        "--alpha", type=float,
        default=2.0,
        help="Threshold for LSD"
    )

    parser.add_argument(
        "--beta", type=float,
        default=-2.0,
        help="Threshold for LCL"
    )
    

    main(parser.parse_args())