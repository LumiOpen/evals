from jinja2 import Template
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
        config = {}
        config["env_vars"] = env_vars
        config["slurm_config"] = slurm_config

        # Select template based on backend
        if backend == 'vllm':
            template_file = 'templates/lm_eval_harness_vllm.sh'
        else:  # hf or auto (defaults to hf)
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

        # BigCode harness currently only supports hf backend
        # TODO: Add vLLM support for BigCode evaluations if needed
        template_str = open('templates/bigcode_eval_harness.sh', 'r').read()
        template = Template(template_str)
        rendered_script = template.render(**config)

        return rendered_script
