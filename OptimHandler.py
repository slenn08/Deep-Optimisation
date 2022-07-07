from typing import Tuple
import torch

from abc import abstractmethod, ABC

from COProblems.OptimizationProblem import OptimizationProblem
from Models.DOBase import DOBase

class OptimHandler(ABC):
    """
    Abstract class to handle the DO algorithm, given a model and a problem. Each handler
    is specific to the model as different model types may require slightly different algorithms.
    """
    def __init__(self, model : DOBase, problem : OptimizationProblem):
        """
        Constructor method for the OptimHandler class.

        Args:
            model: DOBase
                The central model being used for DO.
            problem: OptimizationProblem
                The combinatorial optimization problem being solved.
        """
        self.model = model
        self.problem = problem
    
    @abstractmethod
    def learn_from_population(self, solutions : torch.Tensor,
                              optimizer : torch.optim.Optimizer, batch_size : int,
                              *args, **kwargs) -> None:
        """
        Method to make the model learn from the population of solutions.

        Args:
            solutions: torch.Tensor
                The solutions to learn from. Has shape N x W, where N is the number of solutions
                in the population and W is the size of each solution.
            optimizer: torch.optim.Optimizer
                The optimizer used to adjust the weights of the model.
            batch_size: int
                The batch size used during the learning process.
        """
        pass

    @abstractmethod
    def optimise_solutions(self, solutions : torch.Tensor, fitnesses : torch.Tensor, change_tolerance : int,
                           *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
        """
        Optimises the solutions using Model-Informed Variation. 

        Args:
            solutions: torch.Tensor
                The solutions to learn from. Has shape N x W, where N is the number of solutions
                in the population and W is the size of each solution.
            fitnesses: torch.Tensor
                The list of fitnesses relating to each solution. Has shape N, where the ith fitness
                corresponds to the ith solution in the solutions tensor.
            change_tolerance: int
                Defines how many neutral or negative fitness changes can be made in a row before a 
                solution is deemed an optima during the optimisation process.
        
        Returns:
            A list containing the optimised solutions, their respective fitnesses, the number of
            evaluations used during the process, and a boolean that is true if one of the solutions
            is a global optima.
        """
        pass