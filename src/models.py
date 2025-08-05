from __future__ import annotations
import numpy as np
import scipy.stats as st
import abc

class Model(abc.ABC):

    def __init__(self) -> None:
        self._w = None
        super().__init__()

    @abc.abstractmethod
    def predict(self, X: np.array) -> np.ndarray:
        """Implement the predict method"""

    @abc.abstractmethod
    def gradient(self, X: np.array, y: np.array) -> np.ndarray:
        """Implement the gradient method"""

    @abc.abstractmethod
    def hessian(self, X: np.array, y: np.array) -> np.ndarray:
        """Implement the hessian method"""

    @abc.abstractmethod
    def error(self, X, y):
        """Implement the error method""" # def __update_error(self, X, y):

    @property
    def w(self):
        return self._w
    
    @w.setter
    def w(self, value):
        self._w = value


class LinearModel(Model):
    def predict(self, X: np.array) -> np.ndarray:
        return np.dot(X, self.w)
    
    def gradient(self, X: np.array, y: np.array) -> np.ndarray:
        return (2/len(X)) * (np.linalg.multi_dot([X.T, X, self.w]) - np.dot(X.T, y))
    
    def hessian(self, X: np.array, y: np.array) -> np.ndarray:
        return (2/len(X)) * np.dot(X.T, X)
    
    def error(self, X, y):
        yhat = self.predict(X)
        errors = yhat - y
        rmse = 1.0/len(X) * np.square(np.linalg.norm(errors))
        return rmse
    
class LogisticModel(Model):
    def predict(self, X: np.array) -> np.ndarray:
        return 1/(1 + np.exp(-(X@self.w)))
    
    def gradient(self, X: np.array, y: np.array) -> np.ndarray:
        return X.T@(self.predict(X) - y)
    
    def hessian(self, X: np.array, y: np.array) -> np.ndarray:
        return (X.T@self.predict(X))@X
    
    def error(self, X, y):
        return np.sum(-y * np.log(self.predict(X)) - (1 - y) * np.log(1 - self.predict(X)))