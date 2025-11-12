from jinja2 import Template, Environment, FileSystemLoader
import copy
import datetime


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

class HELMETHarness:
    def __init__(self, config_name):
        self.config_name = config_name
        self.harness = self

    def generate_script(self, slurm_config, env_vars, backend='hf'):
        env_vars = copy.deepcopy(env_vars)
        env_vars["CONFIG_NAME"] = self.config_name
        env_vars["BACKEND"] = backend
        config = {}
        config["env_vars"] = env_vars
        config["slurm_config"] = slurm_config

        template_str = open('templates/helmet.sh', 'r').read()
        template = Template(template_str)
        rendered_script = template.render(**config)

        return rendered_script

class LongPPLHarness:
    def __init__(self, context_length, dataset_samples=50, alpha=2.0, beta=-2.0):
        self.context_length = context_length
        self.dataset_samples = dataset_samples
        self.alpha = alpha
        self.beta = beta
        self.harness = self

    def generate_script(self, slurm_config, env_vars, backend='hf'):
        # LongPPL only supports HF backend (no vLLM support)
        if backend == 'vllm':
            print("Warning: LongPPL does not support vLLM backend, using HF instead")
            backend = 'hf'

        env_vars = copy.deepcopy(env_vars)
        env_vars["CONTEXT_LENGTH"] = self.context_length
        env_vars["DATASET_SAMPLES"] = self.dataset_samples
        env_vars["ALPHA"] = self.alpha
        env_vars["BETA"] = self.beta
        env_vars["BACKEND"] = backend

        config = {}
        config["env_vars"] = env_vars
        config["slurm_config"] = slurm_config

        env = Environment(loader=FileSystemLoader('templates'))
        template = env.get_template('longppl.sh')
        rendered_script = template.render(**config)

        return rendered_script
