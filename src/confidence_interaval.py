import numpy as np
import scipy.stats as st

class ConfidenceInterval:

    def __init__(self, data, alpha):
        self.data = data
        self.alpha = alpha

    @property
    def mean(self):
        return np.mean(self.data)
    
    @property
    def std(self):
        return np.std(self.data)
    
    @property
    def t(self):
        return st.t.ppf(1 - self.alpha/2, len(self.data) - 1)
    
    def calculate_lower_bound(self):
        return self.mean - self.t * (self.std/np.sqrt(len(self.data)))
    
    def calculate_upper_bound(self):
        return self.mean + self.t * (self.std/np.sqrt(len(self.data)))