from abc import ABC, abstractmethod
from .harnesses import LMEvalHarness, BigcodeEvaluationHarness, HELMETHarness
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
        super().__init__(name, result_type, LMEvalHarness([name], num_fewshot=num_fewshot))
        self.num_fewshot = num_fewshot

class MultiMetricLMEvalConfig(EvalConfig):
    def __init__(self, name, metrics, num_fewshot=0):
        super().__init__(name, "custom", LMEvalHarness([name], num_fewshot=num_fewshot))
        self.metrics = metrics

    def get_results_custom(self, json_data):
        task_results = json_data.get("results", {}).get(self.name)
        if task_results is None:
            return None

        parsed_results = {}
        for metric in self.metrics:
            metric_key = next(
                (
                    key
                    for key in (f"{metric},none", metric)
                    if key in task_results
                ),
                None,
            )
            if metric_key is not None:
                parsed_results[metric] = task_results[metric_key]

            stderr_key = next(
                (
                    key
                    for key in (
                        f"{metric}_stderr,none",
                        f"{metric}_stderr",
                    )
                    if key in task_results
                ),
                None,
            )
            if stderr_key is not None:
                parsed_results[f"{metric}_stderr"] = task_results[stderr_key]

        if "samples" in task_results:
            parsed_results["samples"] = task_results["samples"]

        return parsed_results

class FinBenchConfig(EvalConfig):
    def __init__(self, name, result_type, num_fewshot=0):
        super().__init__(name, result_type, LMEvalHarness([name], num_fewshot=num_fewshot))
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

class HELMETConfig(EvalConfig):
    def __init__(self, name, config_name):
        super().__init__(name, "custom", HELMETHarness(config_name))
        self.config_name = config_name

    def get_results_custom(self, json_data):
        # HELMET saves results to a directory with multiple files
        # The main summary is in the json_data we load
        return json_data


evals = {
    # finbench
    "finbench_0shot": FinBenchConfig("finbench_multiple_choice", "custom", num_fewshot=0),
    "finbench_1shot": FinBenchConfig("finbench_multiple_choice", "custom", num_fewshot=1),
    "finbench_2shot": FinBenchConfig("finbench_multiple_choice", "custom", num_fewshot=2),
    "finbench_3shot": FinBenchConfig("finbench_multiple_choice", "custom", num_fewshot=3),

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

    # english score through same task config
    "gsm8k_mt_en_baseline": LMEvalConfig("ogx_gsm8kx", "acc,none", num_fewshot=5),
    "gsm8k_mt_bg": LMEvalConfig("ogx_gsm8kx_bg", "acc,none", num_fewshot=5),
    "gsm8k_mt_cs": LMEvalConfig("ogx_gsm8kx_cs", "acc,none", num_fewshot=5),
    "gsm8k_mt_da": LMEvalConfig("ogx_gsm8kx_da", "acc,none", num_fewshot=5),
    "gsm8k_mt_de": LMEvalConfig("ogx_gsm8kx_de", "acc,none", num_fewshot=5),
    "gsm8k_mt_el": LMEvalConfig("ogx_gsm8kx_el", "acc,none", num_fewshot=5),
    "gsm8k_mt_es": LMEvalConfig("ogx_gsm8kx_es", "acc,none", num_fewshot=5),
    "gsm8k_mt_et": LMEvalConfig("ogx_gsm8kx_et", "acc,none", num_fewshot=5),
    "gsm8k_mt_fi": LMEvalConfig("ogx_gsm8kx_fi", "acc,none", num_fewshot=5),
    "gsm8k_mt_fr": LMEvalConfig("ogx_gsm8kx_fr", "acc,none", num_fewshot=5),
    "gsm8k_mt_hu": LMEvalConfig("ogx_gsm8kx_hu", "acc,none", num_fewshot=5),
    "gsm8k_mt_it": LMEvalConfig("ogx_gsm8kx_it", "acc,none", num_fewshot=5),
    "gsm8k_mt_lt": LMEvalConfig("ogx_gsm8kx_lt", "acc,none", num_fewshot=5),
    "gsm8k_mt_lv": LMEvalConfig("ogx_gsm8kx_lv", "acc,none", num_fewshot=5),
    "gsm8k_mt_nl": LMEvalConfig("ogx_gsm8kx_nl", "acc,none", num_fewshot=5),
    "gsm8k_mt_pl": LMEvalConfig("ogx_gsm8kx_pl", "acc,none", num_fewshot=5),
    "gsm8k_mt_pt": LMEvalConfig("ogx_gsm8kx_pt-pt", "acc,none", num_fewshot=5),
    "gsm8k_mt_ro": LMEvalConfig("ogx_gsm8kx_ro", "acc,none", num_fewshot=5),
    "gsm8k_mt_sk": LMEvalConfig("ogx_gsm8kx_sk", "acc,none", num_fewshot=5),
    "gsm8k_mt_sl": LMEvalConfig("ogx_gsm8kx_sl", "acc,none", num_fewshot=5),
    "gsm8k_mt_sv": LMEvalConfig("ogx_gsm8kx_sv", "acc,none", num_fewshot=5),

    "hellaswag_mt_bg": LMEvalConfig("ogx_hellaswagx_bg", "acc,norm", num_fewshot=10),
    "hellaswag_mt_cs": LMEvalConfig("ogx_hellaswagx_cs", "acc,norm", num_fewshot=10),
    "hellaswag_mt_da": LMEvalConfig("ogx_hellaswagx_da", "acc,norm", num_fewshot=10),
    "hellaswag_mt_de": LMEvalConfig("ogx_hellaswagx_de", "acc,norm", num_fewshot=10),
    "hellaswag_mt_el": LMEvalConfig("ogx_hellaswagx_el", "acc,norm", num_fewshot=10),
    "hellaswag_mt_es": LMEvalConfig("ogx_hellaswagx_es", "acc,norm", num_fewshot=10),
    "hellaswag_mt_et": LMEvalConfig("ogx_hellaswagx_et", "acc,norm", num_fewshot=10),
    "hellaswag_mt_fi": LMEvalConfig("ogx_hellaswagx_fi", "acc,norm", num_fewshot=10),
    "hellaswag_mt_fr": LMEvalConfig("ogx_hellaswagx_fr", "acc,norm", num_fewshot=10),
    "hellaswag_mt_hu": LMEvalConfig("ogx_hellaswagx_hu", "acc,norm", num_fewshot=10),
    "hellaswag_mt_it": LMEvalConfig("ogx_hellaswagx_it", "acc,norm", num_fewshot=10),
    "hellaswag_mt_lt": LMEvalConfig("ogx_hellaswagx_lt", "acc,norm", num_fewshot=10),
    "hellaswag_mt_lv": LMEvalConfig("ogx_hellaswagx_lv", "acc,norm", num_fewshot=10),
    "hellaswag_mt_nl": LMEvalConfig("ogx_hellaswagx_nl", "acc,norm", num_fewshot=10),
    "hellaswag_mt_pl": LMEvalConfig("ogx_hellaswagx_pl", "acc,norm", num_fewshot=10),
    "hellaswag_mt_pt": LMEvalConfig("ogx_hellaswagx_pt-pt", "acc,norm", num_fewshot=10),
    "hellaswag_mt_ro": LMEvalConfig("ogx_hellaswagx_ro", "acc,norm", num_fewshot=10),
    "hellaswag_mt_sk": LMEvalConfig("ogx_hellaswagx_sk", "acc,norm", num_fewshot=10),
    "hellaswag_mt_sl": LMEvalConfig("ogx_hellaswagx_sl", "acc,norm", num_fewshot=10),
    "hellaswag_mt_sv": LMEvalConfig("ogx_hellaswagx_sv", "acc,norm", num_fewshot=10),

    "goldenswag_mt_bg": LMEvalConfig("ogx_goldenswagx_bg", "acc,norm", num_fewshot=0),
    "goldenswag_mt_cs": LMEvalConfig("ogx_goldenswagx_cs", "acc,norm", num_fewshot=0),
    "goldenswag_mt_da": LMEvalConfig("ogx_goldenswagx_da", "acc,norm", num_fewshot=0),
    "goldenswag_mt_de": LMEvalConfig("ogx_goldenswagx_de", "acc,norm", num_fewshot=0),
    "goldenswag_mt_el": LMEvalConfig("ogx_goldenswagx_el", "acc,norm", num_fewshot=0),
    "goldenswag_mt_es": LMEvalConfig("ogx_goldenswagx_es", "acc,norm", num_fewshot=0),
    "goldenswag_mt_et": LMEvalConfig("ogx_goldenswagx_et", "acc,norm", num_fewshot=0),
    "goldenswag_mt_fi": LMEvalConfig("ogx_goldenswagx_fi", "acc,norm", num_fewshot=0),
    "goldenswag_mt_fr": LMEvalConfig("ogx_goldenswagx_fr", "acc,norm", num_fewshot=0),
    "goldenswag_mt_hu": LMEvalConfig("ogx_goldenswagx_hu", "acc,norm", num_fewshot=0),
    "goldenswag_mt_it": LMEvalConfig("ogx_goldenswagx_it", "acc,norm", num_fewshot=0),
    "goldenswag_mt_lt": LMEvalConfig("ogx_goldenswagx_lt", "acc,norm", num_fewshot=0),
    "goldenswag_mt_lv": LMEvalConfig("ogx_goldenswagx_lv", "acc,norm", num_fewshot=0),
    "goldenswag_mt_nl": LMEvalConfig("ogx_goldenswagx_nl", "acc,norm", num_fewshot=0),
    "goldenswag_mt_pl": LMEvalConfig("ogx_goldenswagx_pl", "acc,norm", num_fewshot=0),
    "goldenswag_mt_pt": LMEvalConfig("ogx_goldenswagx_pt-pt", "acc,norm", num_fewshot=0),
    "goldenswag_mt_ro": LMEvalConfig("ogx_goldenswagx_ro", "acc,norm", num_fewshot=0),
    "goldenswag_mt_sk": LMEvalConfig("ogx_goldenswagx_sk", "acc,norm", num_fewshot=0),
    "goldenswag_mt_sl": LMEvalConfig("ogx_goldenswagx_sl", "acc,norm", num_fewshot=0),
    "goldenswag_mt_sv": LMEvalConfig("ogx_goldenswagx_sv", "acc,norm", num_fewshot=0),

    "goldenswag_mt_bg_10shot": LMEvalConfig("ogx_goldenswagx_bg", "acc,norm", num_fewshot=10),
    "goldenswag_mt_cs_10shot": LMEvalConfig("ogx_goldenswagx_cs", "acc,norm", num_fewshot=10),
    "goldenswag_mt_da_10shot": LMEvalConfig("ogx_goldenswagx_da", "acc,norm", num_fewshot=10),
    "goldenswag_mt_de_10shot": LMEvalConfig("ogx_goldenswagx_de", "acc,norm", num_fewshot=10),
    "goldenswag_mt_el_10shot": LMEvalConfig("ogx_goldenswagx_el", "acc,norm", num_fewshot=10),
    "goldenswag_mt_es_10shot": LMEvalConfig("ogx_goldenswagx_es", "acc,norm", num_fewshot=10),
    "goldenswag_mt_et_10shot": LMEvalConfig("ogx_goldenswagx_et", "acc,norm", num_fewshot=10),
    "goldenswag_mt_fi_10shot": LMEvalConfig("ogx_goldenswagx_fi", "acc,norm", num_fewshot=10),
    "goldenswag_mt_fr_10shot": LMEvalConfig("ogx_goldenswagx_fr", "acc,norm", num_fewshot=10),
    "goldenswag_mt_hu_10shot": LMEvalConfig("ogx_goldenswagx_hu", "acc,norm", num_fewshot=10),
    "goldenswag_mt_it_10shot": LMEvalConfig("ogx_goldenswagx_it", "acc,norm", num_fewshot=10),
    "goldenswag_mt_lt_10shot": LMEvalConfig("ogx_goldenswagx_lt", "acc,norm", num_fewshot=10),
    "goldenswag_mt_lv_10shot": LMEvalConfig("ogx_goldenswagx_lv", "acc,norm", num_fewshot=10),
    "goldenswag_mt_nl_10shot": LMEvalConfig("ogx_goldenswagx_nl", "acc,norm", num_fewshot=10),
    "goldenswag_mt_pl_10shot": LMEvalConfig("ogx_goldenswagx_pl", "acc,norm", num_fewshot=10),
    "goldenswag_mt_pt_10shot": LMEvalConfig("ogx_goldenswagx_pt-pt", "acc,norm", num_fewshot=10),
    "goldenswag_mt_ro_10shot": LMEvalConfig("ogx_goldenswagx_ro", "acc,norm", num_fewshot=10),
    "goldenswag_mt_sk_10shot": LMEvalConfig("ogx_goldenswagx_sk", "acc,norm", num_fewshot=10),
    "goldenswag_mt_sl_10shot": LMEvalConfig("ogx_goldenswagx_sl", "acc,norm", num_fewshot=10),
    "goldenswag_mt_sv_10shot": LMEvalConfig("ogx_goldenswagx_sv", "acc,norm", num_fewshot=10),


    "truthfulqa_mc_mt_bg": LMEvalConfig("ogx_truthfulqax_mc2_bg", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_cs": LMEvalConfig("ogx_truthfulqax_mc2_cs", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_da": LMEvalConfig("ogx_truthfulqax_mc2_da", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_de": LMEvalConfig("ogx_truthfulqax_mc2_de", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_el": LMEvalConfig("ogx_truthfulqax_mc2_el", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_es": LMEvalConfig("ogx_truthfulqax_mc2_es", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_et": LMEvalConfig("ogx_truthfulqax_mc2_et", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_fi": LMEvalConfig("ogx_truthfulqax_mc2_fi", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_fr": LMEvalConfig("ogx_truthfulqax_mc2_fr", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_hu": LMEvalConfig("ogx_truthfulqax_mc2_hu", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_it": LMEvalConfig("ogx_truthfulqax_mc2_it", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_lt": LMEvalConfig("ogx_truthfulqax_mc2_lt", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_lv": LMEvalConfig("ogx_truthfulqax_mc2_lv", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_nl": LMEvalConfig("ogx_truthfulqax_mc2_nl", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_pl": LMEvalConfig("ogx_truthfulqax_mc2_pl", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_pt": LMEvalConfig("ogx_truthfulqax_mc2_pt-pt", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_ro": LMEvalConfig("ogx_truthfulqax_mc2_ro", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_sk": LMEvalConfig("ogx_truthfulqax_mc2_sk", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_sl": LMEvalConfig("ogx_truthfulqax_mc2_sl", "acc,none", num_fewshot=5),
    "truthfulqa_mc_mt_sv": LMEvalConfig("ogx_truthfulqax_mc2_sv", "acc,none", num_fewshot=5),

    "belebele_eng": LMEvalConfig("belebele_eng_Latn", "acc,none", num_fewshot=5),
    "belebele_dan": LMEvalConfig("belebele_dan_Latn", "acc,none", num_fewshot=5),
    "belebele_fin": LMEvalConfig("belebele_fin_Latn", "acc,none", num_fewshot=5),
    "belebele_isl": LMEvalConfig("belebele_isl_Latn", "acc,none", num_fewshot=5),
    "belebele_nno": LMEvalConfig("belebele_nno_Latn", "acc,none", num_fewshot=5),
    "belebele_nob": LMEvalConfig("belebele_nob_Latn", "acc,none", num_fewshot=5),
    "belebele_swe": LMEvalConfig("belebele_swe_Latn", "acc,none", num_fewshot=5),

    "flores200_trans_en_da": LMEvalConfig("ogx_flores200-trans-eng_Latn-dan_Latn", "bleu_flores200,none", num_fewshot=8),
    "flores200_trans_en_fi": LMEvalConfig("ogx_flores200-trans-eng_Latn-fin_Latn", "bleu_flores200,none", num_fewshot=8),
    "flores200_trans_en_is": LMEvalConfig("ogx_flores200-trans-eng_Latn-isl_Latn", "bleu_flores200,none", num_fewshot=8),
    "flores200_trans_en_nb": LMEvalConfig("ogx_flores200-trans-eng_Latn-nob_Latn", "bleu_flores200,none", num_fewshot=8),
    "flores200_trans_en_sv": LMEvalConfig("ogx_flores200-trans-eng_Latn-swe_Latn", "bleu_flores200,none", num_fewshot=8),
    "flores200_trans_da_en": LMEvalConfig("ogx_flores200-trans-dan_Latn-eng_Latn", "bleu_flores200,none", num_fewshot=8),
    "flores200_trans_fi_en": LMEvalConfig("ogx_flores200-trans-fin_Latn-eng_Latn", "bleu_flores200,none", num_fewshot=8),
    "flores200_trans_is_en": LMEvalConfig("ogx_flores200-trans-isl_Latn-eng_Latn", "bleu_flores200,none", num_fewshot=8),
    "flores200_trans_nb_en": LMEvalConfig("ogx_flores200-trans-nob_Latn-eng_Latn", "bleu_flores200,none", num_fewshot=8),
    "flores200_trans_sv_en": LMEvalConfig("ogx_flores200-trans-swe_Latn-eng_Latn", "bleu_flores200,none", num_fewshot=8),

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

    "goldenswag": LMEvalConfig("goldenswag", "acc_norm,none", num_fewshot=0),
    "goldenswag_10shot": LMEvalConfig("goldenswag", "acc_norm,none", num_fewshot=10),

    # code
    "humaneval_pass@1": BigcodeConfig("humaneval", n_samples=1),
    "humaneval_pass@10": BigcodeConfig("humaneval", n_samples=10),
    "humaneval-unstripped_pass@1": BigcodeConfig("humaneval-unstripped", n_samples=1),
    "humaneval-unstripped_pass@10": BigcodeConfig("humaneval-unstripped", n_samples=10),
    "mbpp_pass@1": BigcodeConfig("mbpp", n_samples=1),
    "mbpp_pass@10": BigcodeConfig("mbpp", n_samples=10),

    # IFEval
    "ifeval": LMEvalConfig("ifeval", "acc,none", num_fewshot=0),
    "ifeval_fi": LMEvalConfig("ifeval_fi", "acc,none", num_fewshot=0),
    "ifeval_sv": LMEvalConfig("ifeval_sv", "acc,none", num_fewshot=0),

    # INCLUDE  
    "include_finnish": LMEvalConfig("include_base_44_finnish", "acc,none", num_fewshot=5),

    # MATH 4-shot  
    "minerva_math": LMEvalConfig("minerva_math", "exact_match,none", num_fewshot=4),

    # harder evals
    "gpqa_diamond_zeroshot": LMEvalConfig("gpqa_diamond_zeroshot", "acc,none", num_fewshot=0),
    "gpqa_diamond_5shot": LMEvalConfig("gpqa_diamond_nshot", "acc,none", num_fewshot=5),
    "gpqa_diamond_cot_zeroshot": LMEvalConfig("gpqa_diamond_cot_zeroshot", "acc,none", num_fewshot=0),
    "gpqa_diamond_cot_5shot": LMEvalConfig("gpqa_diamond_cot_nshot", "acc,none", num_fewshot=5),

    # HELMET long-context evaluations (standalone framework)
    # Recall tasks
    "helmet_recall": HELMETConfig("helmet_recall", "recall"),
    "helmet_recall_short": HELMETConfig("helmet_recall_short", "recall_short"),
    "helmet_recall_vllm": HELMETConfig("helmet_recall_vllm", "recall_vllm"),
    "helmet_niah": HELMETConfig("helmet_niah", "niah"),
    "helmet_niah_long": HELMETConfig("helmet_niah_long", "niah_long"),
    "helmet_ruler": HELMETConfig("helmet_ruler", "ruler"),
    "helmet_ruler_short": HELMETConfig("helmet_ruler_short", "ruler_short"),

    # RAG tasks
    "helmet_rag": HELMETConfig("helmet_rag", "rag"),
    "helmet_rag_short": HELMETConfig("helmet_rag_short", "rag_short"),
    "helmet_rag_vllm": HELMETConfig("helmet_rag_vllm", "rag_vllm"),

    # Reranking tasks
    "helmet_rerank": HELMETConfig("helmet_rerank", "rerank"),
    "helmet_rerank_short": HELMETConfig("helmet_rerank_short", "rerank_short"),

    # Citation tasks
    "helmet_cite": HELMETConfig("helmet_cite", "cite"),
    "helmet_cite_short": HELMETConfig("helmet_cite_short", "cite_short"),
    "helmet_alce_nocite": HELMETConfig("helmet_alce_nocite", "alce_nocite"),
    "helmet_alce_nocite_short": HELMETConfig("helmet_alce_nocite_short", "alce_nocite_short"),

    # Long QA tasks
    "helmet_longqa": HELMETConfig("helmet_longqa", "longqa"),
    "helmet_longqa_short": HELMETConfig("helmet_longqa_short", "longqa_short"),

    # Summarization tasks
    "helmet_summ": HELMETConfig("helmet_summ", "summ"),
    "helmet_summ_short": HELMETConfig("helmet_summ_short", "summ_short"),

    # In-context learning tasks
    "helmet_icl": HELMETConfig("helmet_icl", "icl"),
    "helmet_icl_short": HELMETConfig("helmet_icl_short", "icl_short"),

    # Demo config for testing
    "helmet_recall_demo": HELMETConfig("helmet_recall_demo", "recall_demo"),
}
