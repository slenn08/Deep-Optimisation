from abc import ABC, abstractmethod

import torch

class OptimisationProblem(ABC):
    """
    Abstract class to represent an optimisation problem. Any problem to solve via DO should
    subclass from it.
    """
    def __init__(self, device=torch.device("cpu")):
        """
        Constructor method for OptimisationProblem.
        """
        self.max_fitness = float('inf')
        self.device = device

    @abstractmethod
    def fitness(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the fitnesses of solutions.

        Args:
            x: torch.Tensor
                The solutions that will have their fitness calculated.
        
        Returns:
            The fitnesses of the solutions.
        """
        pass

    @abstractmethod
    def is_valid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Determines whether the given solutions violates any constraints on the problem.

        Args:
            x: torch.Tensor
                The solutions which will be tested.

        Returns:
            A tensor where the ith element will be True if solution i is valid, and False otherwise.
        """
        pass

    @abstractmethod
    def random_solution(self, pop_size: int) -> torch.Tensor:
        """
        Generates a population of solutions to the problem.

        Returns:
            The random solutions.
        """
        pass

    @abstractmethod    
    def repair(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repairs solutions so that they remain valid.

        Args:
            x: torch.Tensor
                The solutions to be repaired.
        
        Returns:
            The repaired solutions.
        """
        pass

