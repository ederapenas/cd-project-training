from __future__ import annotations
from typing import List
import src.algorithms as al
import abc

class StopCriteria(abc.ABC):

    @abc.abstractmethod
    def isFinished(self, alg: al.Algorithm) -> bool:
        """Implement stop criterium"""

class CompositeStopCriteria(StopCriteria):

    children: List[StopCriteria]
    def __init__(self) -> None:
        super().__init__()
        self.children = []

    def add(self, stop_criteria: StopCriteria):
        self.children.append(stop_criteria)

    def isFinished(self, alg: al.Algorithm) -> bool:
        return any([s.isFinished(alg) for s in self.children])
    
class MaxIterationsStopCriteria(StopCriteria):
    def __init__(self, max_iterations) -> None:
        super().__init__()
        self.max_iterations = max_iterations

    def isFinished(self, alg: al.Algorithm) -> bool:
        return alg.iteration >= self.max_iterations
    
class MinErrorStopCriteria(StopCriteria):
    def __init__(self, min_error) -> None:
        super().__init__()
        self.min_error = min_error

    def isFinished(self, alg: al.Algorithm) -> bool:
        if not alg.errors:
            return False
        return min(alg.errors) <= self.min_error