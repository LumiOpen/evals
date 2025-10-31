#!/usr/bin/env python3
"""Generate per-dataset HELMET config files"""
import os
from pathlib import Path

# RAG config data
rag_datasets = {
    'nq': {
        'name': 'kilt_nq',
        'k_values': {
            '8k': 50, '16k': 105, '32k': 220, '64k': 440, '128k': 1000
        },
        'dep': 6,
        'test_file_template': 'data/kilt/nq-dev-multikilt_1000_k{k}_dep6.jsonl',
        'demo_file': 'data/kilt/nq-train-multikilt_1000_k3_dep6.jsonl'
    },
    'triviaqa': {
        'name': 'kilt_triviaqa',
        'k_values': {
            '8k': 50, '16k': 105, '32k': 220, '64k': 440, '128k': 1000
        },
        'dep': 6,
        'test_file_template': 'data/kilt/triviaqa-dev-multikilt_1000_k{k}_dep6.jsonl',
        'demo_file': 'data/kilt/triviaqa-train-multikilt_1000_k3_dep6.jsonl'
    },
    'hotpotqa': {
        'name': 'kilt_hotpotqa',
        'k_values': {
            '8k': 50, '16k': 105, '32k': 220, '64k': 440, '128k': 1000
        },
        'dep': 3,
        'test_file_template': 'data/kilt/hotpotqa-dev-multikilt_1000_k{k}_dep3.jsonl',
        'demo_file': 'data/kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl'
    },
    'popqa': {
        'name': 'kilt_popqa_3',
        'k_values': {
            '8k': 50, '16k': 105, '32k': 220, '64k': 440, '128k': 1000
        },
        'dep': 6,
        'test_file_template': 'data/kilt/popqa_test_1000_k{k}_dep6.jsonl',
        'demo_file': 'data/kilt/popqa_test_1000_k3_dep6.jsonl'
    }
}

# Longqa config data
longqa_datasets = {
    'narrativeqa': {
        'names': {
            '8k': 'narrativeqa_7892',
            '16k': 'narrativeqa_16084',
            '32k': 'narrativeqa_32468',
            '64k': 'narrativeqa_64836',
            '128k': 'narrativeqa_130604'
        },
        'generation_max_length': 100
    },
    'infbench_qa': {
        'names': {
            '8k': 'infbench_qa_eng_7982',
            '16k': 'infbench_qa_eng_16174',
            '32k': 'infbench_qa_eng_32558',
            '64k': 'infbench_qa_eng_64926',
            '128k': 'infbench_qa_eng_130694'
        },
        'generation_max_length': 10
    },
    'infbench_choice': {
        'names': {
            '8k': 'infbench_choice_eng_7982',
            '16k': 'infbench_choice_eng_16174',
            '32k': 'infbench_choice_eng_32558',
            '64k': 'infbench_choice_eng_64926',
            '128k': 'infbench_choice_eng_130694'
        },
        'generation_max_length': 10
    }
}

context_lengths = {
    '8k': 8192,
    '16k': 16384,
    '32k': 32768,
    '64k': 65536,
    '128k': 131072
}

def main():
    output_dir = Path('helmet/configs')
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files = []

    # Generate RAG configs
    print("Generating RAG configs...")
    for dataset_key, params in rag_datasets.items():
        for ctx in ['8k', '16k', '32k', '64k', '128k']:
            k = params['k_values'][ctx]
            max_len = context_lengths[ctx]
            filename = f"rag_{dataset_key}_{ctx}.yaml"
            filepath = output_dir / filename
            test_file = params['test_file_template'].format(k=k)

            config = f"""input_max_length: {max_len}
datasets: {params['name']}
generation_max_length: 20
test_files: {test_file}
demo_files: {params['demo_file']}
use_chat_template: false
shots: 2
stop_new_line: true
max_test_samples: 2000
"""
            filepath.write_text(config)
            created_files.append(str(filepath))
            print(f"  Created {filename}")

    # Generate longqa configs
    print("\nGenerating longqa configs...")
    for dataset_key, params in longqa_datasets.items():
        for ctx in ['8k', '16k', '32k', '64k', '128k']:
            max_len = context_lengths[ctx]
            dataset_name = params['names'][ctx]
            gen_len = params['generation_max_length']
            filename = f"longqa_{dataset_key}_{ctx}.yaml"
            filepath = output_dir / filename

            config = f"""input_max_length: {max_len}
datasets: {dataset_name}
generation_max_length: {gen_len}
test_files: ','
demo_files: ','
use_chat_template: true
shots: 2
stop_new_line: false
"""
            filepath.write_text(config)
            created_files.append(str(filepath))
            print(f"  Created {filename}")

    print(f"\n✅ Created {len(created_files)} config files in {output_dir}/")
    return created_files

if __name__ == '__main__':
    main()
