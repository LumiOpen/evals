from jinja2 import Template
import argparse
import copy
import datetime
import json
import os
import re
import subprocess
import tempfile

class FinBench:
    def __init__(self, num_fewshot=0):
        self.num_fewshot = num_fewshot

    def generate_script(self, slurm_config, env_vars):
        env_vars = copy.deepcopy(env_vars)
        env_vars["NUM_FEWSHOT"] = self.num_fewshot
        config = {}
        config["env_vars"] = env_vars
        config["slurm_config"] = slurm_config

        template_str = open('templates/finbench.sh', 'r').read()
        template = Template(template_str)
        rendered_script = template.render(**config)

        return rendered_script

class LMEvalHarness:
    def __init__(self, task_list, num_fewshot=0):
        self.task_list = task_list
        self.num_fewshot = num_fewshot

    def generate_script(self, slurm_config, env_vars):
        env_vars = copy.deepcopy(env_vars)
        env_vars["TASK_LIST"] = ",".join(self.task_list)
        env_vars["NUM_FEWSHOT"] = self.num_fewshot
        config = {}
        config["env_vars"] = env_vars
        config["slurm_config"] = slurm_config

        template_str = open('templates/lm_eval_harness.sh', 'r').read()
        template = Template(template_str)
        rendered_script = template.render(**config)

        return rendered_script

class BigcodeEvaluationHarness:
    def __init__(self, task_list, n_samples=1):
        self.task_list = task_list
        self.n_samples = n_samples

    def generate_script(self, slurm_config, env_vars):
        env_vars = copy.deepcopy(env_vars)
        env_vars["TASK_LIST"] = ",".join(self.task_list)
        env_vars["N_SAMPLES"] = self.n_samples
        config = {}
        config["env_vars"] = env_vars
        config["slurm_config"] = slurm_config

        template_str = open('templates/bigcode_eval_harness.sh', 'r').read()
        template = Template(template_str)
        rendered_script = template.render(**config)

        return rendered_script

evals = {
    "finbench_0shot": FinBench(num_fewshot=0),
    "finbench_1shot": FinBench(num_fewshot=1),
    "finbench_2shot": FinBench(num_fewshot=2),
    "finbench_3shot": FinBench(num_fewshot=3),

    # These are all configured as in the HF leaderboard for easy
    # comparison.
    "arc_challenge": LMEvalHarness(["arc_challenge"], num_fewshot=25),
    "hellaswag": LMEvalHarness(["hellaswag"], num_fewshot=10),
    "mmlu": LMEvalHarness(["hendrycksTest-*"], num_fewshot=5),
    "truthfulqa_mc": LMEvalHarness(["truthfulqa_mc"], num_fewshot=0),
    "winogrande": LMEvalHarness(["winogrande"], num_fewshot=5),
    "gsm8k": LMEvalHarness(["gsm8k"], num_fewshot=5),
    "drop": LMEvalHarness(["drop"], num_fewshot=3),

    "arc_easy": LMEvalHarness(["arc_easy"], num_fewshot=0),
    "boolq": LMEvalHarness(["boolq"], num_fewshot=0),
    "nq_open": LMEvalHarness(["nq_open"], num_fewshot=0),
    "openbookqa": LMEvalHarness(["openbookqa"], num_fewshot=0),
    "piqa": LMEvalHarness(["piqa"], num_fewshot=0),
    "race": LMEvalHarness(["race"], num_fewshot=0),
    "triviaqa": LMEvalHarness(["triviaqa"], num_fewshot=0),

    # TODO probably specify different max durations here for longer running
    # tests. I'm not even sure we can complete a 100 sample test.
    "humaneval_pass@1": BigcodeEvaluationHarness(["humaneval"], n_samples=1),
    "humaneval_pass@10": BigcodeEvaluationHarness(["humaneval"], n_samples=10),
    "humaneval_pass@100": BigcodeEvaluationHarness(["humaneval"], n_samples=100),
}


def run_eval(eval_name, args):
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    output_dir = args.output_dir
    if not output_dir:
        # determine output dir
        output_dir = "./output"

        # parse the model if we can do so
        if "viking_v2" in args.model:
            # viking_v2_7B_iter_0096000_bfloat16
            result = re.search(r"(viking_v2_\d+B)_iter_(\d+)_bfloat16", args.model)
            if result:
                model_name = result.group(1)
                step = result.group(2)
                output_dir = os.path.join(output_dir, f"{model_name}/{step}")
        elif "/scratch/project_462000319/general-tools/checkpoints/" in args.model:
            # sample model is "33B_torch_step67824_bfloat16"
            model_info_search = re.search(r"([^/]+)_torch_(.+)_bfloat16", args.model)
            if model_info_search:
                model_name = model_info_search.group(1)
                step = model_info_search.group(2)
                if model_name == "33B":  # this labeling is ambihguous, fix it.
                    model_name = "poro-34b"
                output_dir = os.path.join(output_dir, f"{model_name}/{step}")
        # match anything of the form "Org/Model" as used on huggingface.
        elif re.match(r"^[^/]+/[^/]+$", args.model):
            output_dir = os.path.join(output_dir, args.model)
        else:
            output_dir = os.path.join(output_dir, "other", os.path.basename(args.model))


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
    script = tester.generate_script(slurm_config, env_vars)

    script_name = args.script_name
    if script_name is not None:
        with open(script_name, 'w') as f:
            f.write(script)
    else:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
            temp.write(script)
            script_name = temp.name
    if args.dryrun:
        print(f"Wrote script to {script_name}")
        print(f"Dryrun mode enabled, not executing.  To run:\n    sbatch {script_name}")
        return


    process = subprocess.run(['sbatch', script_name], capture_output=True, text=True)
    # parse the jobid from stdout
    job_id_search = re.search(r"Submitted batch job (\d+)", process.stdout)
    if job_id_search:
        job_id = job_id_search.group(1)
    else:
        print("Failed to parse job id from sbatch output:")
        print(process)
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
    parser.add_argument('--work_dir', type=str, required=False, default="./workdir")
    parser.add_argument('--trust_remote_code', action='store_true', default=False, help="load model with trust_remote_code=True")

    # slurm config
    parser.add_argument('--project', type=str, default="project_462000086", help="Project for sbatch job")
    parser.add_argument('--partition', type=str, default="small-g", help="Partition for sbatch job")
    parser.add_argument('--gres', type=str, default="gpu:mi250:2", help="gres required for sbatch job")
    parser.add_argument('--time', type=str, default="48:00:00", help="Time limit for sbatch job")
    parser.add_argument('--log_dir', type=str, default="./logs", help="Dir for slurm logs")

    # other options
    parser.add_argument('--comment', type=str, default=None, help="Comment to add to the command history")
    parser.add_argument('--script_name', type=str, default=None, help="Filename to use when writing script.")
    parser.add_argument('--dryrun', action='store_true', default=False, help="Dry run mode, do not execute sbatch script")
    parser.add_argument('--force', action='store_true', default=False, help="Run even if output file already exists")

    args = parser.parse_args()
    if args.tokenizer == "":
        args.tokenizer = args.model

    for eval_name in args.eval:
        run_eval(eval_name, args)

if __name__ == "__main__":
    main()
