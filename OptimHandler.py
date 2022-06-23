from typing import Tuple
import torch

from abc import abstractmethod, ABC

from .COProblems import OptimizationProblem
from .Models.DOBase import DOBase

class OptimHandler(ABC):
    def __init__(self, model : DOBase, problem : OptimizationProblem):
        self.model = model
        self.problem = problem
    
    @abstractmethod
    def learn_from_population(solutions : torch.Tensor,
                              optimizer : torch.optim.Optimizer, batch_size : int,
                              *args, **kwargs) -> None:
        pass

    @abstractmethod
    def optimise_solutions(solutions : torch.Tensor, fitnesses : torch.Tensor,
                           *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
        pass