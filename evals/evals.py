from abc import ABC, abstractmethod
from .harnesses import LMEvalHarness, BigcodeEvaluationHarness, MTBenchInferenceHarness, MTBenchJudgeHarness, AlpacaEvalHarness, MultiIFHarness, EuroEvalHarness
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

class MTBenchInferenceConfig(EvalConfig):
    def __init__(self, language="en"):
        super().__init__("mtbench_inference", "custom", MTBenchInferenceHarness(language=language))
        self.language = language

    def get_results_custom(self, json_data):
        # MTBench inference doesn't produce final results, just generates answers
        return None

class MTBenchJudgeConfig(EvalConfig):
    def __init__(self, language="en", judge_model="gpt-4o-2024-08-06"):
        super().__init__("mtbench_judge", "custom", MTBenchJudgeHarness(language=language, judge_model=judge_model))
        self.language = language
        self.judge_model = judge_model

    def get_results_custom(self, json_data):
        # Extract the average score from MTBench results
        if "results" in json_data and "first_turn_score" in json_data["results"]:
            return json_data["results"]["first_turn_score"]
        return 0.0

class AlpacaEvalConfig(EvalConfig):
    def __init__(self, language="en", annotator_config="weighted_alpaca_eval_gpt-4o-2024-08-06"):
        super().__init__("alpaca_eval", "custom", AlpacaEvalHarness(language=language, annotator_config=annotator_config))
        self.language = language
        self.annotator_config = annotator_config

    def get_results_custom(self, json_data):
        # Extract win rate from AlpacaEval results
        if "results" in json_data and "length_controlled_winrate" in json_data["results"]:
            return json_data["results"]["length_controlled_winrate"]
        return 0.0

class MultiIFConfig(EvalConfig):
    def __init__(self, batch_size=64, tensor_parallel_size=8, input_data_csv="data/multiIF_20241018_english.csv"):
        super().__init__("multi_if", "custom", MultiIFHarness(
            batch_size=batch_size,
            tensor_parallel_size=tensor_parallel_size,
            input_data_csv=input_data_csv)
        )
        self.batch_size = batch_size
        self.tensor_parallel_size = tensor_parallel_size
        self.input_data_csv = input_data_csv

    def get_results_custom(self, json_data):
        # Extract overall score from Multi-IF results
        if "results" in json_data and "english_average" in json_data["results"]:
            return json_data["results"]["english_average"]
        return 0.0


class EuroEvalConfig(EvalConfig):
    """Config for EuroEval European language benchmarks.

    EuroEval supports multiple European languages and task types:
    - Languages: da, sv, no, fi, is, de, nl, en, fr, es, it, pt, pl, etc.
    - Tasks: sentiment-classification, named-entity-recognition,
             linguistic-acceptability, reading-comprehension,
             knowledge, summarization, common-sense-reasoning
    """
    def __init__(self, name, language=None, task=None, dataset=None,
                 few_shot=True, evaluate_test_split=False):
        super().__init__(name, "custom", EuroEvalHarness(
            language=language,
            task=task,
            dataset=dataset,
            few_shot=few_shot,
            evaluate_test_split=evaluate_test_split,
        ))
        self.language = language
        self.task = task
        self.dataset = dataset
        self.few_shot = few_shot
        self.evaluate_test_split = evaluate_test_split

    def get_results_custom(self, json_data):
        # EuroEval outputs results as JSONL, parse accordingly
        # The results contain task scores per language
        if isinstance(json_data, list):
            # JSONL format - get the last result entry
            if json_data:
                last_result = json_data[-1]
                if "results" in last_result:
                    # Return the mean score across all tasks
                    scores = []
                    for task_results in last_result["results"].values():
                        if "total" in task_results:
                            scores.append(task_results["total"])
                    if scores:
                        return sum(scores) / len(scores)
        return 0.0



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

    # post-training evaluations
    "mtbench_inference_en": MTBenchInferenceConfig(language="en"),
    "mtbench_inference_fi": MTBenchInferenceConfig(language="fi"),
    "mtbench_judge_en": MTBenchJudgeConfig(language="en"),
    "mtbench_judge_fi": MTBenchJudgeConfig(language="fi"),
    "alpaca_eval_en": AlpacaEvalConfig(language="en"),
    "alpaca_eval_fi": AlpacaEvalConfig(language="fi"),
    "multi_if": MultiIFConfig(),

    # =========================================================================
    # EuroEval - European Language Model Benchmarks
    # https://euroeval.com/
    # =========================================================================

    # Full language evaluations (all tasks for a language)
    "euroeval_da": EuroEvalConfig("euroeval_da", language="da"),
    "euroeval_sv": EuroEvalConfig("euroeval_sv", language="sv"),
    "euroeval_no": EuroEvalConfig("euroeval_no", language="no"),
    "euroeval_fi": EuroEvalConfig("euroeval_fi", language="fi"),
    "euroeval_is": EuroEvalConfig("euroeval_is", language="is"),
    "euroeval_de": EuroEvalConfig("euroeval_de", language="de"),
    "euroeval_nl": EuroEvalConfig("euroeval_nl", language="nl"),
    "euroeval_en": EuroEvalConfig("euroeval_en", language="en"),
    "euroeval_fr": EuroEvalConfig("euroeval_fr", language="fr"),
    "euroeval_es": EuroEvalConfig("euroeval_es", language="es"),
    "euroeval_it": EuroEvalConfig("euroeval_it", language="it"),
    "euroeval_pt": EuroEvalConfig("euroeval_pt", language="pt"),
    "euroeval_pl": EuroEvalConfig("euroeval_pl", language="pl"),
    "euroeval_cs": EuroEvalConfig("euroeval_cs", language="cs"),
    "euroeval_sk": EuroEvalConfig("euroeval_sk", language="sk"),
    "euroeval_hu": EuroEvalConfig("euroeval_hu", language="hu"),
    "euroeval_ro": EuroEvalConfig("euroeval_ro", language="ro"),
    "euroeval_bg": EuroEvalConfig("euroeval_bg", language="bg"),
    "euroeval_el": EuroEvalConfig("euroeval_el", language="el"),
    "euroeval_et": EuroEvalConfig("euroeval_et", language="et"),
    "euroeval_lv": EuroEvalConfig("euroeval_lv", language="lv"),
    "euroeval_lt": EuroEvalConfig("euroeval_lt", language="lt"),
    "euroeval_sl": EuroEvalConfig("euroeval_sl", language="sl"),
    "euroeval_hr": EuroEvalConfig("euroeval_hr", language="hr"),
    "euroeval_uk": EuroEvalConfig("euroeval_uk", language="uk"),

    # Task-specific evaluations (all languages for a task)
    "euroeval_sentiment": EuroEvalConfig("euroeval_sentiment", task="sentiment-classification"),
    "euroeval_ner": EuroEvalConfig("euroeval_ner", task="named-entity-recognition"),
    "euroeval_la": EuroEvalConfig("euroeval_la", task="linguistic-acceptability"),
    "euroeval_rc": EuroEvalConfig("euroeval_rc", task="reading-comprehension"),
    "euroeval_know": EuroEvalConfig("euroeval_know", task="knowledge"),
    "euroeval_summ": EuroEvalConfig("euroeval_summ", task="summarization"),
    "euroeval_reason": EuroEvalConfig("euroeval_reason", task="common-sense-reasoning"),

    # Nordic languages bundle
    "euroeval_nordic": EuroEvalConfig("euroeval_nordic", language="da,sv,no,fi,is"),

    # Zero-shot variants
    "euroeval_da_0shot": EuroEvalConfig("euroeval_da_0shot", language="da", few_shot=False),
    "euroeval_sv_0shot": EuroEvalConfig("euroeval_sv_0shot", language="sv", few_shot=False),
    "euroeval_fi_0shot": EuroEvalConfig("euroeval_fi_0shot", language="fi", few_shot=False),
    "euroeval_en_0shot": EuroEvalConfig("euroeval_en_0shot", language="en", few_shot=False),
    "euroeval_de_0shot": EuroEvalConfig("euroeval_de_0shot", language="de", few_shot=False),
}
