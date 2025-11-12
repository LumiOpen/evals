#!/usr/bin/env python3
"""Parse LongPPL log output and create JSON results file."""
import argparse
import re
import json
import sys
import os
from datetime import datetime


def parse_longppl_log(log_file, model_id, context_length, dataset_samples, alpha, beta, slurm_job_id):
    """Parse LongPPL log and extract metrics."""
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    with open(log_file, errors='ignore') as f:
        log_text = f.read()

    # Extract longppl and ppl values from output
    # Format: "model_name: longppl: XX.XX, ppl: YY.YY"
    # Match the final line with both values
    final_line_match = re.search(r"longppl:\s*([\d.]+),\s*ppl:\s*([\d.]+)", log_text)

    if not final_line_match:
        print("Warning: Could not parse longppl/ppl from output")
        print("Log excerpt:")
        print(log_text[-500:])
        longppl_val = None
        ppl_val = None
    else:
        longppl_val = float(final_line_match.group(1))
        ppl_val = float(final_line_match.group(2))

    return {
        "model": model_id,
        "context_length": context_length,
        "dataset": "govreport-test-tokenized",
        "dataset_samples": dataset_samples,
        "alpha": alpha,
        "beta": beta,
        "longppl": longppl_val,
        "ppl": ppl_val,
        "timestamp": datetime.now().isoformat(),
        "slurm_job_id": slurm_job_id
    }


def main():
    parser = argparse.ArgumentParser(description="Parse LongPPL results from log file")
    parser.add_argument('--log-file', required=True, help='Path to LongPPL log file')
    parser.add_argument('--output-file', required=True, help='Path to output JSON file')
    parser.add_argument('--model-id', required=True, help='Model identifier')
    parser.add_argument('--context-length', type=int, required=True, help='Context length used')
    parser.add_argument('--dataset-samples', type=int, required=True, help='Number of dataset samples')
    parser.add_argument('--alpha', type=float, required=True, help='Alpha threshold')
    parser.add_argument('--beta', type=float, required=True, help='Beta threshold')
    parser.add_argument('--slurm-job-id', default='', help='SLURM job ID')

    args = parser.parse_args()

    results = parse_longppl_log(
        args.log_file, args.model_id, args.context_length,
        args.dataset_samples, args.alpha, args.beta, args.slurm_job_id
    )

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {args.output_file}")
    if results['longppl'] is not None:
        print(f"LongPPL: {results['longppl']:.4f}, PPL: {results['ppl']:.4f}")
    else:
        print("Warning: Could not parse results from log")


if __name__ == '__main__':
    main()
