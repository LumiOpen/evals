from pathlib import Path

import datasets

from lm_eval.api.task import ConfigurableTask, TaskConfig
from lm_eval.tasks.helmet import utils as helmet_utils

DATA_FILE = Path(__file__).parent / "data" / "helmet_cite_local.jsonl"


class LocalHelmetCiteTask(ConfigurableTask):
    """Lightweight HELMET citation task backed by a tiny on-disk dataset."""

    VERSION = 0
    OUTPUT_TYPE = "generate_until"
    CONFIG = TaskConfig(
        task="helmet_cite_local",
        dataset_path=None,
        dataset_name=None,
        training_split=None,
        validation_split=None,
        test_split="train",
        process_docs=None,
        doc_to_text=lambda doc: doc["input"],
        doc_to_target=lambda doc: doc["target"],
        num_fewshot=0,
        metric_list=None,
        generation_kwargs={
            "max_gen_toks": 32,
            "temperature": 0.0,
            "do_sample": False,
            "until": ["\n\n"],
        },
    )

    def download(self, data_dir=None, cache_dir=None, download_mode=None) -> None:
        dataset = datasets.load_dataset(
            "json", data_files=str(DATA_FILE), cache_dir=cache_dir
        )
        processed = helmet_utils.process_cite_docs(dataset["train"])
        self.dataset = datasets.DatasetDict({"train": processed})

    def process_results(self, doc, results):
        prediction = results[0]
        target = doc["target"]
        exact = 1.0 if prediction.strip() == (target or "").strip() else 0.0
        return {"exact_match": exact}

    def aggregation(self):
        return {"exact_match": lambda items: sum(items) / len(items)}

    def higher_is_better(self):
        return {"exact_match": True}
