#!/usr/bin/env python3
"""
Convert JSON files to HuggingFace Dataset format for faster loading.
Usage: python convert_json_to_hf_dataset.py input.json output_dir
"""

import sys
import json
from datasets import Dataset

def convert_json_to_dataset(json_path, output_dir):
    print(f"Loading JSON from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents")

    # Convert to HuggingFace Dataset
    print("Converting to HuggingFace Dataset...")
    dataset = Dataset.from_list(data)

    print(f"Saving to {output_dir}...")
    dataset.save_to_disk(output_dir)

    print(f"✓ Conversion complete!")
    print(f"  Input:  {json_path}")
    print(f"  Output: {output_dir}")
    print(f"  Documents: {len(dataset)}")

    # Verify by loading
    print("\nVerifying by reloading...")
    from datasets import load_from_disk
    test_dataset = load_from_disk(output_dir)
    print(f"✓ Verification successful: {len(test_dataset)} documents")
    print(f"  Sample keys: {list(test_dataset[0].keys())}")
    if 'text' in test_dataset[0]:
        print(f"  Sample text length: {len(test_dataset[0]['text'])} chars")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_json_to_hf_dataset.py input.json output_dir")
        sys.exit(1)

    json_path = sys.argv[1]
    output_dir = sys.argv[2]

    convert_json_to_dataset(json_path, output_dir)
