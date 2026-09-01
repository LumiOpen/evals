import copy
import os
from pathlib import Path
import shlex

from jinja2 import Environment


TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def render_template(name, config):
    environment = Environment(autoescape=False, keep_trailing_newline=True)
    environment.filters["shellquote"] = shlex.quote
    template = environment.from_string((TEMPLATES_DIR / name).read_text())
    return template.render(**config)


class LMEvalHarness:
    def __init__(self, task_list, num_fewshot=0):
        self.task_list = task_list
        self.num_fewshot = num_fewshot
        self.harness = self # TODO remove

    def generate_script(self, slurm_config, env_vars, backend='hf'):
        env_vars = copy.deepcopy(env_vars)
        env_vars["TASK_LIST"] = ",".join(self.task_list)
        env_vars["NUM_FEWSHOT"] = self.num_fewshot
        env_vars["BACKEND"] = backend
        config = {}
        config["env_vars"] = env_vars
        config["slurm_config"] = slurm_config

        return render_template('lm_eval_harness.sh', config)

class BigcodeEvaluationHarness:
    def __init__(self, task_list, n_samples=1):
        self.task_list = task_list
        self.n_samples = n_samples
        self.harness = self

    def generate_script(self, slurm_config, env_vars, backend='hf'):
        env_vars = copy.deepcopy(env_vars)
        env_vars["TASK_LIST"] = ",".join(self.task_list)
        env_vars["N_SAMPLES"] = self.n_samples
        config = {}
        config["env_vars"] = env_vars
        config["slurm_config"] = slurm_config

        return render_template('bigcode_eval_harness.sh', config)

class MTBenchInferenceHarness:
    def __init__(self, language="en"):
        self.language = language
        self.harness = self

    def generate_script(self, slurm_config, env_vars):
        env_vars = copy.deepcopy(env_vars)
        model_id = os.path.basename(env_vars["MODEL"])
        env_vars["ANSWER_FILE"] = f"data/mt_bench/model_answer/{model_id}-{self.language}.jsonl"
        env_vars["LANGUAGE"] = self.language
        config = {}
        config["env_vars"] = env_vars
        config["slurm_config"] = slurm_config

        return render_template('mtbench_inference.sh', config)

class MTBenchJudgeHarness:
    def __init__(self, language="en", judge_model="gpt-4o-2024-08-06"):
        self.language = language
        self.judge_model = judge_model
        self.harness = self

    def generate_script(self, slurm_config, env_vars):
        # Check if OPENAI_API_KEY is set since this harness uses GPT judge models
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("MTBenchJudgeHarness requires OPENAI_API_KEY environment variable to be set for GPT judge model")

        env_vars = copy.deepcopy(env_vars)
        env_vars["OPENAI_API_KEY"] = openai_api_key
        env_vars["LANGUAGE"] = self.language
        env_vars["JUDGE_MODEL"] = self.judge_model
        config = {}
        config["env_vars"] = env_vars
        config["slurm_config"] = slurm_config

        return render_template('mtbench_judge.sh', config)

class AlpacaEvalHarness:
    def __init__(self, language="en", annotator_config="weighted_alpaca_eval_gpt-4o-2024-08-06"):
        self.language = language
        self.annotator_config = annotator_config
        self.harness = self

    def generate_script(self, slurm_config, env_vars):
        # Check if OPENAI env vars are set since this harness uses GPT annotator models
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("AlpacaEvalHarness requires OPENAI_API_KEY environment variable to be set for GPT annotator model")
        
        openai_org_id = os.environ.get("OPENAI_ORG_ID")
        if not openai_org_id:
            raise ValueError("AlpacaEvalHarness requires OPENAI_ORG_ID environment variable to be set for GPT annotator model")
                
        env_vars = copy.deepcopy(env_vars)
        env_vars["OPENAI_API_KEY"] = openai_api_key
        env_vars["OPENAI_ORG_ID"] = openai_org_id
        env_vars["LANGUAGE"] = self.language
        env_vars["ANNOTATOR_CONFIG"] = self.annotator_config
        model_id = os.path.basename(env_vars["MODEL"])
        env_vars["RESULTS_DIR"] = f"results/{model_id}-{self.language}"
        config = {}
        config["env_vars"] = env_vars
        config["slurm_config"] = slurm_config

        return render_template('alpaca_eval.sh', config)

class MultiIFHarness:
    def __init__(self, batch_size=64, tensor_parallel_size=8, input_data_csv="data/multiIF_20241018_english.csv"):
        self.batch_size = batch_size
        self.tensor_parallel_size = tensor_parallel_size
        self.input_data_csv = input_data_csv
        self.harness = self

    def generate_script(self, slurm_config, env_vars):
        env_vars = copy.deepcopy(env_vars)
        env_vars["BATCH_SIZE"] = self.batch_size
        env_vars["TENSOR_PARALLEL_SIZE"] = self.tensor_parallel_size
        env_vars["INPUT_DATA_CSV"] = self.input_data_csv
        config = {}
        config["env_vars"] = env_vars
        config["slurm_config"] = slurm_config

        return render_template('multi_if.sh', config)

