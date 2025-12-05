from jinja2 import Template
import copy
import datetime
import os


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


class EuroEvalHarness:
    """Harness for running EuroEval benchmarks on LUMI.

    EuroEval is a European language model benchmarking framework supporting
    12+ European languages with tasks like sentiment classification, NER,
    reading comprehension, and more.

    See: https://euroeval.com/
    """
    def __init__(self, language=None, task=None, dataset=None, few_shot=True,
                 evaluate_test_split=False, num_iterations=10, batch_size=None,
                 gpu_memory_utilization=0.85):
        self.language = language
        self.task = task
        self.dataset = dataset
        self.few_shot = few_shot
        self.evaluate_test_split = evaluate_test_split
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.harness = self

    def generate_script(self, slurm_config, env_vars, backend='hf'):
        env_vars = copy.deepcopy(env_vars)

        # Add EuroEval-specific configuration
        if self.language:
            env_vars["LANGUAGE"] = self.language
        if self.task:
            env_vars["TASK"] = self.task
        if self.dataset:
            env_vars["DATASET"] = self.dataset
        env_vars["FEW_SHOT"] = str(self.few_shot)
        env_vars["EVALUATE_TEST_SPLIT"] = str(self.evaluate_test_split)
        env_vars["NUM_ITERATIONS"] = self.num_iterations
        if self.batch_size:
            env_vars["BATCH_SIZE"] = self.batch_size
        env_vars["GPU_MEMORY_UTILIZATION"] = self.gpu_memory_utilization

        config = {}
        config["env_vars"] = env_vars
        config["slurm_config"] = slurm_config

        template_str = open('templates/euroeval.sh', 'r').read()
        template = Template(template_str)
        rendered_script = template.render(**config)

        return rendered_script

