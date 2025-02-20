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

# These configs can be used directly for simple tests, or subclassed for more complex ones.
class LMEvalConfig(EvalConfig):
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
    # finbench
    "finbench_0shot": FinBenchConfig("finbench", "custom", num_fewshot=0),
    "finbench_1shot": FinBenchConfig("finbench", "custom", num_fewshot=1),
    "finbench_2shot": FinBenchConfig("finbench", "custom", num_fewshot=2),
    "finbench_3shot": FinBenchConfig("finbench", "custom", num_fewshot=3),

    # non-english evals
    "arc_challenge_mt_bg": LMEvalConfig("arc_challenge_mt_bg", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_cs": LMEvalConfig("arc_challenge_mt_cs", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_da": LMEvalConfig("arc_challenge_mt_da", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_de": LMEvalConfig("arc_challenge_mt_de", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_el": LMEvalConfig("arc_challenge_mt_el", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_es": LMEvalConfig("arc_challenge_mt_es", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_et": LMEvalConfig("arc_challenge_mt_et", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_fi": LMEvalConfig("arc_challenge_mt_fi", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_fr": LMEvalConfig("arc_challenge_mt_fr", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_hu": LMEvalConfig("arc_challenge_mt_hu", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_is": LMEvalConfig("arc_challenge_mt_is", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_it": LMEvalConfig("arc_challenge_mt_it", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_lt": LMEvalConfig("arc_challenge_mt_lt", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_lv": LMEvalConfig("arc_challenge_mt_lv", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_nb": LMEvalConfig("arc_challenge_mt_nb", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_nl": LMEvalConfig("arc_challenge_mt_nl", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_pl": LMEvalConfig("arc_challenge_mt_pl", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_pt": LMEvalConfig("arc_challenge_mt_pt", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_ro": LMEvalConfig("arc_challenge_mt_ro", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_sk": LMEvalConfig("arc_challenge_mt_sk", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_sl": LMEvalConfig("arc_challenge_mt_sl", "acc_norm", num_fewshot=25),
    "arc_challenge_mt_sv": LMEvalConfig("arc_challenge_mt_sv", "acc_norm", num_fewshot=25),

    "mmlu_mt_bg": LMEvalConfig("ogx_mmlux_BG", "custom", num_fewshot=5),
    "mmlu_mt_cs": LMEvalConfig("ogx_mmlux_CS", "custom", num_fewshot=5),
    "mmlu_mt_da": LMEvalConfig("ogx_mmlux_DA", "custom", num_fewshot=5),
    "mmlu_mt_de": LMEvalConfig("ogx_mmlux_DE", "custom", num_fewshot=5),
    "mmlu_mt_el": LMEvalConfig("ogx_mmlux_EL", "custom", num_fewshot=5),
    "mmlu_mt_es": LMEvalConfig("ogx_mmlux_ES", "custom", num_fewshot=5),
    "mmlu_mt_et": LMEvalConfig("ogx_mmlux_ET", "custom", num_fewshot=5),
    "mmlu_mt_fi": LMEvalConfig("ogx_mmlux_FI", "custom", num_fewshot=5),
    "mmlu_mt_fr": LMEvalConfig("ogx_mmlux_FR", "custom", num_fewshot=5),
    "mmlu_mt_hu": LMEvalConfig("ogx_mmlux_HU", "custom", num_fewshot=5),
    "mmlu_mt_it": LMEvalConfig("ogx_mmlux_IT", "custom", num_fewshot=5),
    "mmlu_mt_lt": LMEvalConfig("ogx_mmlux_LT", "custom", num_fewshot=5),
    "mmlu_mt_lv": LMEvalConfig("ogx_mmlux_LV", "custom", num_fewshot=5),
    "mmlu_mt_nl": LMEvalConfig("ogx_mmlux_NL", "custom", num_fewshot=5),
    "mmlu_mt_pl": LMEvalConfig("ogx_mmlux_PL", "custom", num_fewshot=5),
    "mmlu_mt_pt": LMEvalConfig("ogx_mmlux_PT-PT", "custom", num_fewshot=5),
    "mmlu_mt_ro": LMEvalConfig("ogx_mmlux_RO", "custom", num_fewshot=5),
    "mmlu_mt_sk": LMEvalConfig("ogx_mmlux_SK", "custom", num_fewshot=5),
    "mmlu_mt_sl": LMEvalConfig("ogx_mmlux_SL", "custom", num_fewshot=5),
    "mmlu_mt_sv": LMEvalConfig("ogx_mmlux_SV", "custom", num_fewshot=5),

    # englishs core through same task config
    "gsm8k_mt_en_baseline": LMEvalConfig("ogx_gsm8kx", "acc,none", num_fewshot=5),
    "gsm8k_mt_bg": LMEvalConfig("ogx_gsm8kx_BG", "acc,none", num_fewshot=5),
    "gsm8k_mt_cs": LMEvalConfig("ogx_gsm8kx_CS", "acc,none", num_fewshot=5),
    "gsm8k_mt_da": LMEvalConfig("ogx_gsm8kx_DA", "acc,none", num_fewshot=5),
    "gsm8k_mt_de": LMEvalConfig("ogx_gsm8kx_DE", "acc,none", num_fewshot=5),
    "gsm8k_mt_el": LMEvalConfig("ogx_gsm8kx_EL", "acc,none", num_fewshot=5),
    "gsm8k_mt_es": LMEvalConfig("ogx_gsm8kx_ES", "acc,none", num_fewshot=5),
    "gsm8k_mt_et": LMEvalConfig("ogx_gsm8kx_ET", "acc,none", num_fewshot=5),
    "gsm8k_mt_fi": LMEvalConfig("ogx_gsm8kx_FI", "acc,none", num_fewshot=5),
    "gsm8k_mt_fr": LMEvalConfig("ogx_gsm8kx_FR", "acc,none", num_fewshot=5),
    "gsm8k_mt_hu": LMEvalConfig("ogx_gsm8kx_HU", "acc,none", num_fewshot=5),
    "gsm8k_mt_it": LMEvalConfig("ogx_gsm8kx_IT", "acc,none", num_fewshot=5),
    "gsm8k_mt_lt": LMEvalConfig("ogx_gsm8kx_LT", "acc,none", num_fewshot=5),
    "gsm8k_mt_lv": LMEvalConfig("ogx_gsm8kx_LV", "acc,none", num_fewshot=5),
    "gsm8k_mt_nl": LMEvalConfig("ogx_gsm8kx_NL", "acc,none", num_fewshot=5),
    "gsm8k_mt_pl": LMEvalConfig("ogx_gsm8kx_PL", "acc,none", num_fewshot=5),
    "gsm8k_mt_pt": LMEvalConfig("ogx_gsm8kx_PT-PT", "acc,none", num_fewshot=5),
    "gsm8k_mt_ro": LMEvalConfig("ogx_gsm8kx_RO", "acc,none", num_fewshot=5),
    "gsm8k_mt_sk": LMEvalConfig("ogx_gsm8kx_SK", "acc,none", num_fewshot=5),
    "gsm8k_mt_sl": LMEvalConfig("ogx_gsm8kx_SL", "acc,none", num_fewshot=5),
    "gsm8k_mt_sv": LMEvalConfig("ogx_gsm8kx_SV", "acc,none", num_fewshot=5),

    "belebele_eng": LMEvalConfig("belebele_eng_Latn", "acc,none", num_fewshot=5),
    "belebele_dan": LMEvalConfig("belebele_dan_Latn", "acc,none", num_fewshot=5),
    "belebele_fin": LMEvalConfig("belebele_fin_Latn", "acc,none", num_fewshot=5),
    "belebele_isl": LMEvalConfig("belebele_isl_Latn", "acc,none", num_fewshot=5),
    "belebele_nno": LMEvalConfig("belebele_nno_Latn", "acc,none", num_fewshot=5),
    "belebele_nob": LMEvalConfig("belebele_nob_Latn", "acc,none", num_fewshot=5),
    "belebele_swe": LMEvalConfig("belebele_swe_Latn", "acc,none", num_fewshot=5),

    # hugging face 6
    "arc_challenge": LMEvalConfig("arc_challenge", "acc_norm,none", num_fewshot=25),
    "hellaswag": LMEvalConfig("hellaswag", "acc_norm,none", num_fewshot=10),
    "mmlu": LMEvalConfig("mmlu", "custom", num_fewshot=5),
    "truthfulqa_mc": LMEvalConfig("truthfulqa", "mc2,none", num_fewshot=0),
    "winogrande": LMEvalConfig("winogrande", "acc,none", num_fewshot=5),
    "gsm8k": LMEvalConfig("gsm8k", "acc,none", num_fewshot=5), # llama 2 uses 8 shot in the paper.

    # additional english
    "arc_easy": LMEvalConfig("arc_easy", "acc,none", num_fewshot=0),
    "boolq": LMEvalConfig("boolq", "acc,none", num_fewshot=0),
    "drop": LMEvalConfig("drop", "f1,none", num_fewshot=3),
    "nq_open": LMEvalConfig("nq_open", "em,none", num_fewshot=0),
    "openbookqa": LMEvalConfig("openbookqa", "acc,none", num_fewshot=0),
    "piqa": LMEvalConfig("piqa", "acc,none", num_fewshot=0),
    "squad2": LMEvalConfig("squadv2", "f1,none", num_fewshot=0),
    "triviaqa": LMEvalConfig("triviaqa", "em,none", num_fewshot=0),

    "toxigen": LMEvalConfig("toxigen", "acc", num_fewshot=0),

    "lambada_openai":  LMEvalConfig("lambada_openai", "acc,none", num_fewshot=0),
    "lambada_standard":  LMEvalConfig("lambada_standard", "acc,none", num_fewshot=0),

    # code
    "humaneval_pass@1": BigcodeConfig("humaneval", n_samples=1),
    "humaneval_pass@10": BigcodeConfig("humaneval", n_samples=10),
    "mbpp_pass@1": BigcodeConfig("mbpp", n_samples=1),
    "mbpp_pass@10": BigcodeConfig("mbpp", n_samples=10),

    # IFEval
    "ifeval": LMEvalConfig("ifeval", "acc,none", num_fewshot=0),
    "ifeval_fi": LMEvalConfig("ifeval_fi", "acc,none", num_fewshot=0),
    "ifeval_sv": LMEvalConfig("ifeval_sv", "acc,none", num_fewshot=0),
}
