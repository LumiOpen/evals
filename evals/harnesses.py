from jinja2 import Template
import copy
import datetime
import os


class LMEvalHarness:
    def __init__(self, task_list, num_fewshot=0, metadata=None):
        self.task_list = task_list
        self.num_fewshot = num_fewshot
        self.metadata = metadata or {}
        self.harness = self # TODO remove

    def generate_script(self, slurm_config, env_vars, backend='hf'):
        env_vars = copy.deepcopy(env_vars)
        env_vars["TASK_LIST"] = ",".join(self.task_list)
        env_vars["NUM_FEWSHOT"] = self.num_fewshot
        env_vars["BACKEND"] = backend
        
        # Add metadata for RULER tasks
        if self.metadata:
            import json
            env_vars["TASK_METADATA"] = json.dumps(self.metadata)
            # Extract max_seq_lengths for model args
            if "max_seq_lengths" in self.metadata and self.metadata["max_seq_lengths"]:
                env_vars["MAX_SEQ_LENGTH"] = str(self.metadata["max_seq_lengths"][0])
        else:
            env_vars["TASK_METADATA"] = ""
            env_vars["MAX_SEQ_LENGTH"] = ""
        
        config = {}
        config["env_vars"] = env_vars
        config["slurm_config"] = slurm_config

        template_file = 'templates/lm_eval_harness.sh'

        template_str = open(template_file, 'r').read()
        template = Template(template_str)
        rendered_script = template.render(**config)

        return rendered_script

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

        template_str = open('templates/bigcode_eval_harness.sh', 'r').read()
        template = Template(template_str)
        rendered_script = template.render(**config)

        return rendered_script

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

        template_str = open('templates/mtbench_inference.sh', 'r').read()
        template = Template(template_str)
        rendered_script = template.render(**config)

        return rendered_script

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

        template_str = open('templates/mtbench_judge.sh', 'r').read()
        template = Template(template_str)
        rendered_script = template.render(**config)

        return rendered_script

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

        template_str = open('templates/alpaca_eval.sh', 'r').read()
        template = Template(template_str)
        rendered_script = template.render(**config)

        return rendered_script

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

        template_str = open('templates/multi_if.sh', 'r').read()
        template = Template(template_str)
        rendered_script = template.render(**config)

        return rendered_script

