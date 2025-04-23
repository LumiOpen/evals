import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile

from evals.evals import evals
from evals.slurm import identify_scheduled_tasks

def parse_model(model):
    # default usually applies to uniquely named finetune tests or failures to parse.
    if model[-1] == '/':
        model = model[:-1]
    step = os.path.basename(model)
    model_name = "other"

    # parse the model if we can do so
    if "europa" in model:
        result = re.search(r"(europa_.*)_iter_(\d+)", model)
        if result:
            model_name = result.group(1)
            step = result.group(2)
    elif "viking_v3" in model:
        result = re.search(r"(viking_v3_\d+B.*)_iter_(\d+)_bfloat16", model)
        if result:
            model_name = result.group(1)
            step = result.group(2)
    elif "viking_v2" in model:
        # viking_v2_7B_iter_0096000_bfloat16
        result = re.search(r"(viking_v2_\d+B)_iter_(\d+)_bfloat16", model)
        if result:
            model_name = result.group(1)
            step = result.group(2)
    elif "/scratch/project_462000319/general-tools/checkpoints/" in model:
        # sample model is "33B_torch_step67824_bfloat16"
        model_info_search = re.search(r"([^/]+)_torch_(.+)_bfloat16", model)
        if model_info_search:
            model_name = model_info_search.group(1)
            step = model_info_search.group(2)
            if model_name == "33B":  # this labeling is ambihguous, fix it.
                model_name = "poro-34b"
    elif "_iter_" in model:
        result = re.search(r"([^/]+)_iter_(\d+)", model)
        if result:
            model_name = result.group(1)
            step = result.group(2)
    elif "checkpoint-" in model:
        # output from rahul's CPT lora script
        # e.g. .../cpt-fi-salamandra-base/checkpoint-8000
        result = re.search(r"([^/]+)/checkpoint-(\d+)", model)
        if result:
            model_name = result.group(1)
            step = result.group(2)
    elif "iter_" in model:
        # .../some_model_name/iter_001000"
        result = re.search(r"([^/]+)/iter_(\d+)", model)
        if result:
            model_name = result.group(1)
            step = result.group(2)
    elif "epoch_" in model:
        # .../some_model_name/epoch_1
        result = re.search(r"([^/]+)/epoch_(\d+)", model)
        if result:
            model_name = result.group(1)
            step = result.group(2)
    else:
        # match anything of the form "Org/Model" as used on huggingface.
        result = re.search(r"^([^/]+)/([^/]+)/?$", model)
        if result:
            # this is actually org / model but ...
            model_name = result.group(1)
            step = result.group(2)
            

    return model_name, step


def run_eval(eval_name, args):
    output_dir = args.output_dir
    if not output_dir:
        model, step = parse_model(args.model)
        output_dir = os.path.join(args.output_root, model, step)

    #
    # generate slurm script
    #

    output_file = os.path.join(os.path.abspath(output_dir), f"{eval_name}.json")
    if os.path.exists(output_file) and not args.force:
        print(f"Output file {output_file} already exists and --force not specified, skipping...")
        return


    env_vars = {
        'MODEL': args.model,
        'TOKENIZER': args.tokenizer,
        'OUTPUT_DIR': os.path.abspath(output_dir),
        'WORK_DIR': os.path.abspath(args.work_dir),
        'OUTPUT_FILE': output_file,
        'TRUST_REMOTE_CODE': "True" if args.trust_remote_code else "False",
    }
    if args.apply_chat_template is False:
        env_vars['APPLY_CHAT_TEMPLATE'] = "False"
    elif args.apply_chat_template is True:
        env_vars['APPLY_CHAT_TEMPLATE'] = "True"
    else:
        env_vars['APPLY_CHAT_TEMPLATE'] = args.apply_chat_template

    if args.fewshot_as_multiturn is False:
        env_vars['FEWSHOT_AS_MULTITURN'] = "False"
    elif args.fewshot_as_multiturn is True:
        env_vars['FEWSHOT_AS_MULTITURN'] = "True"
    else:
        env_vars['FEWSHOT_AS_MULTITURN'] = args.apply_chat_template

    slurm_config = {
        'name': eval_name,
        'account': args.project,
        'partition': args.partition,
        'gres': args.gres,
        'time': args.time,
        'log_dir': os.path.abspath(args.log_dir),
    }

    # eval is a reserved keyword, so we'll use tester instead.
    tester = evals[eval_name]
    harness = tester.harness
    script = harness.generate_script(slurm_config, env_vars)

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
        temp.write(script)
        script_name = temp.name

    if args.dryrun:
        print(f"Wrote script to {script_name}")
        print(f"Dryrun mode enabled, not executing.  To run:\n    sbatch {script_name}")
        return


    #
    # slurm job execution
    #

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # process = subprocess.run(['sbatch', script_name], capture_output=True, text=True)
    process = subprocess.run(['sbatch', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stdout = process.stdout
    stderr = process.stderr
    if process.returncode != 0:
        print(f"Error: sbatch command failed with return code {process.returncode}")
        print(process.stderr)
        return

    # parse the jobid from stdout
    job_id_search = re.search(r"Submitted batch job (\d+)", process.stdout)
    if job_id_search:
        job_id = job_id_search.group(1)
    else:
        print("Failed to parse job id from sbatch output:")
        print(process.stderr)
        # we could possibly return here instead of logging this entry, as we do
        # for a return code != 0
        job_id = None

    # save to command log to help figure out which jobs were which commands
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script_name": script_name,
        "job_id": job_id, 
        "eval": eval_name,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "err_log": os.path.join(os.path.abspath(args.log_dir), f"{job_id}.err"),
        "out_log": os.path.join(os.path.abspath(args.log_dir), f"{job_id}.out"),
        "output_file": output_file,
        "comment": args.comment,
    }
    with open("command_history.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(json.dumps(log_entry, indent=4))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('eval', type=str, nargs='+', choices=list(evals.keys()), default='finbench', help='Which eval to run')

    # env vars
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--tokenizer', type=str, required=False, default="", help='Path to the tokenizer')
    parser.add_argument('--output_dir', type=str, required=False, default="")
    parser.add_argument('--output_root', type=str, required=False, default="./output/v2")
    parser.add_argument('--work_dir', type=str, required=False, default="./workdir")
    parser.add_argument('--trust_remote_code', action='store_true', default=False, help="load model with trust_remote_code=True")
    parser.add_argument(
        '--apply_chat_template',
        nargs='?',
        const=True,
        default=False,
        type=str,
        help=(
            "If true, apply the default chat template. "
            "If provided without an argument, applies the default template. "
            "To specificy a particular template, supply its name (see lm_eval options)"
        )
    )
    parser.add_argument(
        '--fewshot_as_multiturn',
        nargs='?',
        const=True,
        default=False,
        type=str,
        help=(
            "If true, treat few-shot prompts as multiturn conversation. "
            "If provided without an argument, applies the default behaviour. "
        )
    )
    # slurm config
    parser.add_argument('--project', type=str, default="project_462000353", help="Project for sbatch job")
    parser.add_argument('--partition', type=str, default="small-g", help="Partition for sbatch job")
    parser.add_argument('--gres', type=str, default="gpu:mi250:4", help="gres required for sbatch job")
    parser.add_argument('--time', type=str, default="48:00:00", help="Time limit for sbatch job")
    parser.add_argument('--log_dir', type=str, default="./logs", help="Dir for slurm logs")

    # other options
    parser.add_argument('--comment', type=str, default=None, help="Comment to add to the command history")
    parser.add_argument('--dryrun', action='store_true', default=False, help="Dry run mode, do not execute sbatch script")
    parser.add_argument('--force', action='store_true', default=False, help="Run even if output file already exists")

    args = parser.parse_args()
    if args.tokenizer == "":
        args.tokenizer = args.model

    # sanity check model and tokenizer before queueing things up
    if args.model.count('/') != 1:
        # not a hugging face model identifier
        if not os.path.exists(os.path.join(args.model, "config.json")):
            print(f"Error: model '{args.model}' looks like a directory, but does not contain a config.json")
            sys.exit(1)
    if args.tokenizer.count('/') != 1:
        if not os.path.exists(os.path.join(args.model, "tokenizer.json")):
            print(f"Error: tokenizer '{args.tokenizer}' looks like a directory, but does not contain a tokenizer.json")
            sys.exit(1)

    # scheduled_tasks = identify_scheduled_tasks()
    for eval_name in args.eval:
        # task = scheduled_tasks.get((args.model, eval_name), None)
        # if task is not None and not args.force:
        #     print(f"Already detected job for {task['eval']} on {task['model']} on slurm job {task['job_id']} and --force not specified, skipping...")
        #     continue
        run_eval(eval_name, args)

if __name__ == "__main__":
    main()
