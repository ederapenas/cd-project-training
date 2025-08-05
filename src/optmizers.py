from tokenize import Double
import numpy as np
import scipy.stats as st
import abc
import src.models as models

class OptmizerStrategy(abc.ABC):

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abc.abstractmethod
    def update_model(self, X, y, model):
        """Implement update weight strategy"""

    class NewtonsMethod(OptmizerStrategy):

        def update_model(self, X, y, model: models.Model):
            model.w = (1 - self.learning_rate) * model.w - self.learning_rate * np.linalg.inv(model.hessian(X, y))@ model.gradient(X, y)

    class SteepestDescent(OptmizerStrategy):

        def update_model(self, X, y, model: models.Model):
            model.w = model.w - self.learning_rate * model.gradient(X, y)