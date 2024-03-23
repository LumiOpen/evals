from abc import ABC, abstractmethod
from .harnesses import FinBench, LMEvalHarness, LMEvalHarness2, BigcodeEvaluationHarness
import json


class EvalConfig(ABC):
    def __init__(self, name, result_type, harness):
        self.name = name
        self.harness = harness
        self.result_type = result_type

    def parse_results(self, results_path):
            with open(results_path, 'r') as f:
                json_data = json.load(f)
            if self.result_type == "custom":
                return self.get_results_custom(json_data)
            return json_data["results"][self.name][result_type]

    def get_results_acc_norm(self, json_data):
        return json_data["results"][self.name]["acc_norm"]
    
    def get_results_acc(self, json_data):
        return json_data["results"][self.name]["acc"]

    def get_results_custom(self, json_data):
        raise NotImplementedError("custom results must be implemented in a subclass")

class LMEvalConfig(EvalConfig):
    def __init__(self, name, result_type, num_fewshot=0):
        # TODO harnesses should be registered and the generate call should just use them to generate a script given a name and a fewshot config
        # rather than being instantiated in every single eval.
        super().__init__(name, result_type, LMEvalHarness([name], num_fewshot=num_fewshot))
        self.num_fewshot = num_fewshot

# These configs can be used directly for simple tests, or subclassed for more complex ones.
class LMEvalConfig2(EvalConfig):
    def __init__(self, name, result_type, num_fewshot=0):
        super().__init__(name, result_type, LMEvalHarness2([name], num_fewshot=num_fewshot))
        self.num_fewshot = num_fewshot

class FinBenchConfig(EvalConfig):
    def __init__(self, name, result_type, num_fewshot=0):
        super().__init__(name, result_type, FinBench(num_fewshot=num_fewshot))
        self.num_fewshot = num_fewshot


evals = {
    "finbench_0shot": FinBenchConfig("finbench", "custom", num_fewshot=0),
    "finbench_1shot": FinBenchConfig("finbench", "custom", num_fewshot=1),
    "finbench_2shot": FinBenchConfig("finbench", "custom", num_fewshot=2),
    "finbench_3shot": FinBenchConfig("finbench", "custom", num_fewshot=3),

    "arc_challenge_fi": LMEvalConfig("arc_challenge_fi", "acc_norm", num_fewshot=25),

    # These are all configured as in the HF leaderboard for easy
    # comparison.
    "arc_challenge": LMEvalConfig("arc_challenge", "acc_norm", num_fewshot=25),
    "hellaswag": LMEvalConfig("hellaswag", "acc_norm", num_fewshot=10),
    "mmlu": LMEvalConfig("hendrycksTest-*", "custom", num_fewshot=5),
    "truthfulqa_mc": LMEvalConfig("truthfulqa_mc", "mc2", num_fewshot=0),
    "winogrande": LMEvalConfig("winogrande", "acc", num_fewshot=5),
    "gsm8k": LMEvalConfig("gsm8k", "acc", num_fewshot=5), # llama 2 uses 8 shot in the paper.

    # TODO: Evalconfigs need to support multiple tasks to convert these correctly.
    # TODO: these don't execute in our time limit, and probably need to be split.
    # TODO: these are broken.
    #"math_1shot": LMEvalHarness([
    #        "math_prealgebra",
    #        "math_algebra",
    #        "math_num_theory",
    #        "math_counting_and_prob",
    #        "math_geometry",
    #        "math_intermediate_algebra",
    #        "math_precalc"
    #    ], num_fewshot=1),

    "arc_easy": LMEvalConfig("arc_easy", "acc", num_fewshot=0),
    "boolq": LMEvalConfig("boolq", "acc", num_fewshot=0),
    "nq_open": LMEvalConfig("nq_open", "em", num_fewshot=0),
    "openbookqa": LMEvalConfig("openbookqa", "acc", num_fewshot=0),
    "piqa": LMEvalConfig("piqa", "acc", num_fewshot=0),
    "squad2": LMEvalConfig("squad2", "f1", num_fewshot=0),
    "triviaqa": LMEvalConfig("triviaqa", "em", num_fewshot=0),
    "triviaqa_5shot": LMEvalConfig("triviaqa_5shot", "em", num_fewshot=5),

    "toxigen": LMEvalConfig("toxigen", "em", num_fewshot=0),

    "humaneval_pass@1": BigcodeEvaluationHarness(["humaneval"], n_samples=1),
    "humaneval_pass@10": BigcodeEvaluationHarness(["humaneval"], n_samples=10),
    "mbpp_pass@1": BigcodeEvaluationHarness(["mbpp"], n_samples=1),
    "mbpp_pass@10": BigcodeEvaluationHarness(["mbpp"], n_samples=10),

    # TODO tests on new harness
    "arc_challenge2": LMEvalConfig2("arc_challenge", "acc_norm", num_fewshot=25),
    "hellaswag2": LMEvalConfig2("hellaswag", "acc_norm", num_fewshot=10),
    "mmlu2": LMEvalConfig2("hendrycksTest-*", "custom", num_fewshot=5),
    "truthfulqa_mc2": LMEvalConfig2("truthfulqa_mc", "mc2", num_fewshot=0),
    "winogrande2": LMEvalConfig2("winogrande", "acc", num_fewshot=5),
    "gsm8k2": LMEvalConfig2("gsm8k", "acc", num_fewshot=5), # llama 2 uses 8 shot in the paper.
}