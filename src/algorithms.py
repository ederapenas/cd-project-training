import numpy as np
import abc
import src.optmizers as opt
import src.models as models
import src.stop_criteria as stop
from src.preprocessing import Preprocessing

class Algorithm(abc.ABC):
    def __init__(self, optmizer_strategy: opt.OptmizerStrategy, model: models.Model) -> None:
        self.algorithm_observers = []
        self.optmizer_strategy = optmizer_strategy
        self.model = model

    def add(self, observer):
        if observer not in self.algorithm_observers:
            self.algorithm_observers.append(observer)
        else:
            print('Failed to add {}', format(observer))

    def remove(self, observer):
        try:
            self.algorithm_observers.remove(observer)
        except ValueError:
            print('Failed to remove {}', format(observer))

    def notify_iteration(self):
        [o.notify_iteration(self) for o in self.algorithm_observers]

    def notify_started(self):
        [o.notify_started(self) for o in self.algorithm_observers]

    def notify_finished(self):
        [o.notify_finished(self) for o in self.algorithm_observers]

    @abc.abstractmethod
    def fit():
        """Implement the fit method"""

    @property
    def iteration(self):
        return self._iteration
    
    @iteration.setter
    def iteration(self, value):
        self._iteration = value

    @property
    def error(self):
        return self._error
    
    @error.setter
    def error(self, value):
        self._error = value

    @property
    def rmse(self):
        return self._rmse
    
    @rmse.setter
    def rmse(self, value):
        self._rmse = value

class PLA(Algorithm):
    def __init__(self, optmizer_strategy: opt.OptmizerStrategy, mode = models.Model):
        super().__init__(optmizer_strategy, mode)
        self.iteration = 0
        self.errors = []
        self.rmse = 2.0

    def fit(self, X, y, stop_criteria: stop.CompositeStopCriteria):
        xones = Preprocessing.build_design_matrix(X)
        self.model.w = np.zeros(xones.shape[1], 1)

        while not stop_criteria.isFinished(self):
            yhat = self.model.predict(xones)
            e = y - yhat
            self.rmse = np.math.sqrt(np.square(e).mean())
            self.optmizer_strategy.update_model(xones, y, self.model)
            self.iteration += 1
            self.errors.append(self.rmse)
            self.notify_iteration()