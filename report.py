import argparse

from evals.evals import evals
from evals.results import Result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add argument for additional arguments that are not attached to flags
    parser.add_argument('results', type=str, nargs='+')
    args = parser.parse_args()

    for result_path in args.results:
        result = Result(result_path)
        print(result.model, result.checkpoint, result.name, result.result)
