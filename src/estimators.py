import numpy as np
import abc

class Estimator(abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def estimate(self, data):
        pass
    
    @property
    def data():
        return self._data
    

class MeanEstimator():
    def estimate(data):
        return np.mean(data)
    

class MedianEstimator():
    def estimate(data):
        return np.median(data)