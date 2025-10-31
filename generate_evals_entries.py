#!/usr/bin/env python3
"""Generate evals.py task registration entries for per-dataset HELMET tasks"""

# Generate RAG per-dataset task registrations
print("# RAG per-dataset tasks")
for dataset in ['nq', 'triviaqa', 'hotpotqa', 'popqa']:
    for ctx in ['8k', '16k', '32k', '64k', '128k']:
        task_name = f"helmet_rag_{dataset}_{ctx}"
        config_name = f"rag_{dataset}_{ctx}"
        print(f'    "{task_name}": HELMETConfig("{task_name}", "{config_name}"),')

print()

# Generate longqa per-dataset task registrations
print("# Longqa per-dataset tasks")
for dataset in ['narrativeqa', 'infbench_qa', 'infbench_choice']:
    for ctx in ['8k', '16k', '32k', '64k', '128k']:
        task_name = f"helmet_longqa_{dataset}_{ctx}"
        config_name = f"longqa_{dataset}_{ctx}"
        print(f'    "{task_name}": HELMETConfig("{task_name}", "{config_name}"),')
