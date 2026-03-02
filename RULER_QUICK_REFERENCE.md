# RULER Quick Reference Guide

## Task Naming Convention

- **Individual tasks**: `ruler_<subtask>_<seqlen>` (e.g., `ruler_niah_single_1_4096`)
- **Grouped tasks**: `ruler_<seqlen>` (e.g., `ruler_4096` runs all 13 subtasks)

## 13 RULER Subtasks

```
NIAH (Needle in a Haystack):
  niah_single_1, niah_single_2, niah_single_3
  niah_multikey_1, niah_multikey_2, niah_multikey_3
  niah_multivalue, niah_multiquery

Other Tasks:
  ruler_vt         - Variable Tracking
  ruler_cwe        - Common Words Extraction
  ruler_fwe        - Frequent Words Extraction
  ruler_qa_hotpot  - Question Answering (Hotpot)
  ruler_qa_squad   - Question Answering (SQuADv2)
```

## 6 Sequence Lengths

```
4096, 8192, 16384, 32768, 65536, 131072
```

## Common Commands

### Test Single Task
```bash
python main.py --model /path/to/model ruler_niah_single_1_4096
```

### Test with Limited Samples
```bash
python main.py --model /path/to/model --limit 10 ruler_niah_single_1_4096
```

### Run Specific Subtasks at Specific Lengths
```bash
sh ruler.sh /path/to/model \
    --subtasks "niah_single_1,ruler_vt" \
    --sequence-lengths "4096,8192"
```

### Run All NIAH Tests
```bash
sh ruler.sh /path/to/model \
    --subtasks "niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery"
```

### Run All Tasks at Short Lengths Only
```bash
sh ruler.sh /path/to/model --sequence-lengths "4096,8192,16384"
```

### Run Everything (78 jobs)
```bash
sh ruler-vllm.sh /path/to/model
```

### With Custom SLURM Config
```bash
sh ruler.sh /path/to/model \
    --partition standard-g \
    --time 24:00:00 \
    --gres gpu:mi250:8
```

## Job Counts

| Configuration | Jobs |
|---------------|------|
| All tasks, all lengths | 13 × 6 = 78 |
| All NIAH, all lengths | 8 × 6 = 48 |
| All tasks, 3 short lengths | 13 × 3 = 39 |
| 2 tasks, 2 lengths | 2 × 2 = 4 |
| Single task | 1 |

## Monitoring

```bash
# Watch jobs
python watch.py

# Check once
python watch.py --once

# View history
python watch.py --hist --days 3

# View RULER results summary
sh summary_ruler.sh output/v2/<model-name>

# View other benchmark results
sh summary.sh output/v2/<model-name>

# List result files
ls output/v2/<model-name>/ruler_*
```

## Backend Options

```bash
# HuggingFace (default)
sh ruler.sh /path/to/model

# vLLM (faster)
sh ruler-vllm.sh /path/to/model

# Manual with specific backend
python main.py --model /path/to/model --backend vllm ruler_niah_single_1_4096
```

## Passing Additional Args

```bash
# Limit samples
sh ruler.sh /path/to/model --limit 50

# Custom lm-eval source
sh ruler.sh /path/to/model --lm_eval https://github.com/EleutherAI/lm-evaluation-harness@main

# Trust remote code
sh ruler.sh /path/to/model --trust_remote_code
```

## Testing Strategy

1. **Start small**: Test 1 task, 1 length, limited samples
   ```bash
   python main.py --model /path/to/model --limit 10 ruler_niah_single_1_4096
   ```

2. **Expand gradually**: Test 2-3 tasks at short lengths
   ```bash
   sh ruler.sh /path/to/model \
       --subtasks "niah_single_1,vt" \
       --sequence-lengths "4096,8192" \
       --limit 50
   ```

3. **Go full scale**: Run all combinations
   ```bash
   sh ruler-vllm.sh /path/to/model
   ```

## Result Files

Results are saved as:
```
output/v2/<model-name>/ruler_<subtask>_<seqlen>.json
output/v2/<model-name>/ruler_<seqlen>.json  (for grouped)
```

Example:
```
output/v2/Llama-3.1-8B/ruler_niah_single_1_4096.json
output/v2/Llama-3.1-8B/ruler_ruler_vt_8192.json
output/v2/Llama-3.1-8B/ruler_ruler_qa_hotpot_16384.json
```

## How It Works

1. Scripts call `python main.py <task_name>`
2. `main.py` generates a SLURM job script
3. `main.py` submits it with `sbatch`
4. SLURM schedules and runs the job
5. Results are saved to output directory

**The scripts do NOT run evaluations locally - they submit them to SLURM!**

