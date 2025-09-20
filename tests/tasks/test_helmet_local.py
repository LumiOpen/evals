from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lm_eval.tasks.helmet.local import LocalHelmetCiteTask


def test_local_helmet_task_limit_and_metrics():
    task = LocalHelmetCiteTask()
    task.set_fewshot_seed(0)

    docs = list(task.doc_iterator(limit=2))
    assert len(docs) == 2

    _, doc = docs[0]
    metrics = task.process_results(doc, ["dummy output"])
    assert "bleu" in metrics
    assert isinstance(metrics["bleu"], float)

    empty_doc = task.dataset["train"][2]
    empty_metrics = task.process_results(empty_doc, ["anything"])
    assert empty_metrics["bleu"] == 0.0
