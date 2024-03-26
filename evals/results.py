import os
from evals.evals import evals


class Result:
    def __init__(self, path):
        self.path = path
        self.model = os.path.basename(os.path.dirname(os.path.dirname(path)))
        self.checkpoint = os.path.basename(os.path.dirname(path))
        self.name = os.path.basename(path).split(".")[0]
    
        self.result = None
        self.load_result()
    
    def load_result(self):
        if self.name not in evals:
            print("Unsupported eval", self.name)
        else:
            self.result = evals[self.name].parse_results(self.path)

    def __str__(self):
        return str(self.result)
    
    def __repr__(self):
        return str(self.result)


