import argparse

from evals.evals import evals
from evals.results import Result


sections = [
    {
        "name": "english core",
        "evals": [
            "arc_challenge.json",
            "gsm8k.json",
            "hellaswag.json",
            "mmlu.json",
            "truthfulqa_mc.json",
            "winogrande.json",
        ],
    }, {
        "name": "finnish",
        "evals": [
            "finbench_3shot.json",
            "arc_challenge_fi.json",
        ],
    }, {
        "name": "nordic other",
        "evals": [
            "arc_challenge_da.json",
            "arc_challenge_nb.json",
            "arc_challenge_sv.json",
        ],
    }, {
        "name": "code",
        "evals": [
            "humaneval_pass@10.json",
            "humaneval_pass@1.json",
            "mbpp_pass@10.json",
            "mbpp_pass@1.json",
        ],
    }, {
        "name": "english extended",
        "evals": [
            "arc_easy.json",
            "boolq.json",
            "nq_open.json",
            "openbookqa.json",
            "piqa.json",
            "squad2.json",
            "triviaqa.json",
        ],
    }, {
        "name": "bias",
        "evals": [
            "toxigen.json",
        ],
    }
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add argument for additional arguments that are not attached to flags
    parser.add_argument('results', type=str, nargs='+')
    args = parser.parse_args()

    for result_path in args.results:
        result = Result(result_path)
        print(result.model, result.checkpoint, result.name, result.result)
