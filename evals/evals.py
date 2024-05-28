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
            return json_data["results"][self.name][self.result_type]

    def get_results_acc_norm(self, json_data):
        return json_data["results"][self.name]["acc_norm"]
    
    def get_results_acc(self, json_data):
        return json_data["results"][self.name]["acc"]

    def get_results_custom(self, json_data):
        return None
        #raise NotImplementedError("custom results must be implemented in a subclass")

class LMEvalConfig(EvalConfig):
    def __init__(self, name, result_type, num_fewshot=0):
        # TODO harnesses should be registered and the generate call should just use them to generate a script given a name and a fewshot config
        # rather than being instantiated in every single eval.
        super().__init__(name, result_type, LMEvalHarness([name], num_fewshot=num_fewshot))
        self.num_fewshot = num_fewshot

    def get_results_custom(self, json_data):
        if self.name == "hendrycksTest-*":
            # avg the acc
            scores = [json_data["results"][task]["acc"] for task in json_data["results"]]
            return sum(scores) / len(scores)
        return None



# These configs can be used directly for simple tests, or subclassed for more complex ones.
class LMEvalConfig2(EvalConfig):
    def __init__(self, name, result_type, num_fewshot=0):
        super().__init__(name, result_type, LMEvalHarness2([name], num_fewshot=num_fewshot))
        self.num_fewshot = num_fewshot

class FinBenchConfig(EvalConfig):
    def __init__(self, name, result_type, num_fewshot=0):
        super().__init__(name, result_type, FinBench(num_fewshot=num_fewshot))
        self.num_fewshot = num_fewshot

    def get_results_custom(self, json_data):
        math_scores = [
            json_data["results"]["bigbench_1_digit_addition"]["multiple_choice_grade"],
            json_data["results"]["bigbench_1_digit_division"]["multiple_choice_grade"],
            json_data["results"]["bigbench_1_digit_multiplication"]["multiple_choice_grade"],
            json_data["results"]["bigbench_1_digit_subtraction"]["multiple_choice_grade"],
            json_data["results"]["bigbench_2_digit_addition"]["multiple_choice_grade"],
            json_data["results"]["bigbench_2_digit_division"]["multiple_choice_grade"],
            json_data["results"]["bigbench_2_digit_multiplication"]["multiple_choice_grade"],
            json_data["results"]["bigbench_2_digit_subtraction"]["multiple_choice_grade"],
            json_data["results"]["bigbench_3_digit_addition"]["multiple_choice_grade"],
            json_data["results"]["bigbench_3_digit_division"]["multiple_choice_grade"],
            json_data["results"]["bigbench_3_digit_multiplication"]["multiple_choice_grade"],
            json_data["results"]["bigbench_3_digit_subtraction"]["multiple_choice_grade"],
            json_data["results"]["bigbench_4_digit_addition"]["multiple_choice_grade"],
            json_data["results"]["bigbench_4_digit_division"]["multiple_choice_grade"],
            json_data["results"]["bigbench_4_digit_multiplication"]["multiple_choice_grade"],
            json_data["results"]["bigbench_4_digit_subtraction"]["multiple_choice_grade"],
            json_data["results"]["bigbench_5_digit_addition"]["multiple_choice_grade"],
            json_data["results"]["bigbench_5_digit_division"]["multiple_choice_grade"],
            json_data["results"]["bigbench_5_digit_multiplication"]["multiple_choice_grade"],
            json_data["results"]["bigbench_5_digit_subtraction"]["multiple_choice_grade"],
        ]
        math_avg = sum(math_scores) / len(math_scores)
        final_scores = [
            json_data["results"]["bigbench_analogies"]["multiple_choice_grade"],
            json_data["results"]["bigbench_emotions"]["multiple_choice_grade"],
            json_data["results"]["bigbench_empirical_judgments"]["multiple_choice_grade"],
            json_data["results"]["bigbench_general_knowledge"]["multiple_choice_grade"],
            json_data["results"]["bigbench_harmless"]["multiple_choice_grade"],
            json_data["results"]["bigbench_helpful"]["multiple_choice_grade"],
            json_data["results"]["bigbench_honest"]["multiple_choice_grade"],
            json_data["results"]["bigbench_intent_recognition"]["multiple_choice_grade"],
            json_data["results"]["bigbench_misconceptions"]["multiple_choice_grade"],
            json_data["results"]["bigbench_one_sentence"]["multiple_choice_grade"],
            json_data["results"]["bigbench_one_sentence_no_prompt"]["multiple_choice_grade"],
            json_data["results"]["bigbench_other"]["multiple_choice_grade"],
            json_data["results"]["bigbench_paraphrase"]["multiple_choice_grade"],
            json_data["results"]["bigbench_sentence_ambiguity"]["multiple_choice_grade"],
            json_data["results"]["bigbench_similarities_abstraction"]["multiple_choice_grade"],
            json_data["results"]["bigbench_two_sentences"]["multiple_choice_grade"], 
            sum(math_scores) / len(math_scores),
        ]
        return sum(final_scores) / len(final_scores)

class BigcodeConfig(EvalConfig):
    def __init__(self, name, n_samples=1):
        super().__init__(name, "custom", BigcodeEvaluationHarness([name], n_samples=n_samples))
        self.n_samples = n_samples

    def get_results_custom(self, json_data):
        return json_data[self.name]["pass@{}".format(self.n_samples)]


evals = {
    "finbench_0shot": FinBenchConfig("finbench", "custom", num_fewshot=0),
    "finbench_1shot": FinBenchConfig("finbench", "custom", num_fewshot=1),
    "finbench_2shot": FinBenchConfig("finbench", "custom", num_fewshot=2),
    "finbench_3shot": FinBenchConfig("finbench", "custom", num_fewshot=3),

    "arc_challenge_da": LMEvalConfig("arc_challenge_da", "acc_norm", num_fewshot=25),
    "arc_challenge_fi": LMEvalConfig("arc_challenge_fi", "acc_norm", num_fewshot=25),
    "arc_challenge_nb": LMEvalConfig("arc_challenge_nb", "acc_norm", num_fewshot=25),
    "arc_challenge_sv": LMEvalConfig("arc_challenge_sv", "acc_norm", num_fewshot=25),
    "arc_challenge_de": LMEvalConfig("arc_challenge_de", "acc_norm", num_fewshot=25),
    "arc_challenge_el": LMEvalConfig("arc_challenge_el", "acc_norm", num_fewshot=25),
    "arc_challenge_es": LMEvalConfig("arc_challenge_es", "acc_norm", num_fewshot=25),
    "arc_challenge_hu": LMEvalConfig("arc_challenge_hu", "acc_norm", num_fewshot=25),
    "arc_challenge_it": LMEvalConfig("arc_challenge_it", "acc_norm", num_fewshot=25),
    "arc_challenge_pl": LMEvalConfig("arc_challenge_pl", "acc_norm", num_fewshot=25),
    "arc_challenge_pt": LMEvalConfig("arc_challenge_pt", "acc_norm", num_fewshot=25),

    "belebele_eng": LMEvalConfig2("belebele_eng_Latn", "acc,none", num_fewshot=5),
    "belebele_dan": LMEvalConfig2("belebele_dan_Latn", "acc,none", num_fewshot=5),
    "belebele_fin": LMEvalConfig2("belebele_fin_Latn", "acc,none", num_fewshot=5),
    "belebele_isl": LMEvalConfig2("belebele_isl_Latn", "acc,none", num_fewshot=5),
    "belebele_nno": LMEvalConfig2("belebele_nno_Latn", "acc,none", num_fewshot=5),
    "belebele_nob": LMEvalConfig2("belebele_nob_Latn", "acc,none", num_fewshot=5),
    "belebele_swe": LMEvalConfig2("belebele_swe_Latn", "acc,none", num_fewshot=5),

    # These are all configured as in the HF leaderboard for easy
    # comparison.
    "arc_challenge": LMEvalConfig("arc_challenge", "acc_norm", num_fewshot=25),
    "hellaswag": LMEvalConfig("hellaswag", "acc_norm", num_fewshot=10),
    "mmlu": LMEvalConfig("hendrycksTest-*", "custom", num_fewshot=5),
    "mmlu_0shot": LMEvalConfig("hendrycksTest-*", "custom", num_fewshot=0),
    "mmlu_1shot": LMEvalConfig("hendrycksTest-*", "custom", num_fewshot=1),
    "mmlu_2shot": LMEvalConfig("hendrycksTest-*", "custom", num_fewshot=2),
    "mmlu_3shot": LMEvalConfig("hendrycksTest-*", "custom", num_fewshot=3),
    "mmlu_4shot": LMEvalConfig("hendrycksTest-*", "custom", num_fewshot=4),
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
    "drop": LMEvalConfig("drop", "f1", num_fewshot=3),
    "nq_open": LMEvalConfig("nq_open", "em", num_fewshot=0),
    "openbookqa": LMEvalConfig("openbookqa", "acc", num_fewshot=0),
    "piqa": LMEvalConfig("piqa", "acc", num_fewshot=0),
    "squad2": LMEvalConfig("squad2", "f1", num_fewshot=0),
    "triviaqa": LMEvalConfig("triviaqa", "em", num_fewshot=0),
    "triviaqa_5shot": LMEvalConfig("triviaqa_5shot", "em", num_fewshot=5),

    "toxigen": LMEvalConfig("toxigen", "acc", num_fewshot=0),

    "humaneval_pass@1": BigcodeConfig("humaneval", n_samples=1),
    "humaneval_pass@10": BigcodeConfig("humaneval", n_samples=10),
    "mbpp_pass@1": BigcodeConfig("mbpp", n_samples=1),
    "mbpp_pass@10": BigcodeConfig("mbpp", n_samples=10),

    # TODO tests on new harness
    "arc_challenge2": LMEvalConfig2("arc_challenge", "acc_norm", num_fewshot=25),
    "hellaswag2": LMEvalConfig2("hellaswag", "acc_norm", num_fewshot=10),
    "mmlu2": LMEvalConfig2("mmlu", "custom", num_fewshot=5),
    "truthfulqa_mc2": LMEvalConfig2("truthfulqa_mc", "mc2", num_fewshot=0),
    "winogrande2": LMEvalConfig2("winogrande", "acc", num_fewshot=5),
    "gsm8k2": LMEvalConfig2("gsm8k", "acc", num_fewshot=5), # llama 2 uses 8 shot in the paper.

    "lambada_openai":  LMEvalConfig("lambada_openai", "acc", num_fewshot=0),
    "lambada_standard":  LMEvalConfig("lambada_standard", "acc", num_fewshot=0),
}
