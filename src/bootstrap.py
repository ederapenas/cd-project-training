import numpy as np

class Bootstrap:

    def __init__(self, data):
        self.data = data
        self.samples = []


    def calculate_bootstrap(self, bootstraps, estimator):
        length = len(self.data)

        for _ in range(bootstraps):
            actual_sample = np.random.choice(self.data, size=length, replace=True)
            self.samples.append(estimator(actual_sample))
    

    @property
    def mean(self):
        return np.mean(self.samples)
    
    @property
    def std(self):
        return np.std(self.samples, ddof=1)