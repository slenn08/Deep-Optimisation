import torch

from COProblems import MKP_populate_function as mkp
from COProblems.OptimisationProblem import OptimisationProblem

class MKP(OptimisationProblem):
    """
    Class to implement the Multi Dimensional Knapsack Problem.
    """
    def __init__(self, file: str, max_fitness_file: str, id: int, device: torch.device):
        """
        Constructor method for MKP. Loads in the weight matrix and constraints matrix for a 
        particular instance of the MKP.

        Args:
            file: str
                The file to extract an MKP instance from.
            max_fitness_file: str
                The file containing the maximum known fitnesses of all the problem instances.
            id: int
                The problem instance ID.
            device: torch.device
                The device that the problem is run on.
        """
        # c = item values
        # A = dimensions x items
        # # Each row is a dimension
        # # Each column is an item
        # b = knapsack size in each dimension
        super().__init__(device)
        self.c, self.A, self.b = mkp.MKPpopulate(file, id)
        self.c = self.c.to(dtype=torch.float32, device=device)
        self.A = self.A.to(dtype=torch.float32, device=device)
        self.b = self.b.to(dtype=torch.float32, device=device)
        self.utility = self.get_utility_order()
        self.max_fitness = mkp.MKPFitness(max_fitness_file, id)
        print("Max possible fitness for this instance: {}".format(self.max_fitness))

    def fitness(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the fitness of any assignment of items.

        Args:
            x: torch.Tensor
                The solutions to have their fitness calculated, where each element is a '1' to 
                represent a 1 and a '-1' to represent a 0.
        
        Returns:
            The fitnesses of each solution.
        """
        x = (x + 1) / 2
        return torch.where(self.is_valid(x), x.matmul(self.c), 0)
    
    def is_valid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Determines whether the given solutions violates any constraints on the problem.

        Args:
            x: torch.Tensor
                The solutions which will be tested.

        Returns:
            A tensor where the ith element will be True if solution i is valid, and False otherwise.
        """
        return (x.matmul(self.A.T) < self.b[None,:]).all(dim=1)

    def random_solution(self, pop_size: int) -> torch.Tensor:
        """
        Generates an empty knapsack for every member of the population.

        Args:
            pop_size: int
                The size of the population of solutions.

        Returns:
            A tensor containing all -1s.
        """
        return torch.full((pop_size, self.c.shape[0]), -1, device=self.device, dtype=torch.float32)
    
    def get_utility_order(self) -> torch.Tensor:
        """
        Calculates the order of the items in terms of their utility.

        Returns:
            A tensor of the items indices in order of utility increasing.
        """
        # Sum columns to get total weight of each item
        total_weight = self.A.sum(axis=0)
        utility = self.c / total_weight
        # Return indices of items in order of utility ascending
        return torch.argsort(utility)
    
    def repair(self, s: torch.Tensor) -> torch.Tensor:
        """
        Repairs invalid solutions to good nearby valid solutions.

        Args:
            s: torch.Tensor   
                The solutions to be repaired.
        
        Returns:
            The repaired solutions.
        """
        s = (s + 1) / 2
        valid = self.is_valid(s)
        # Remove items with low utility until feasible
        for i in self.utility:
            s[:,i] = torch.where(~valid & (s[:,i] == 1), 0, s[:,i])
            valid[~valid] = self.is_valid(s[~valid])
            if torch.all(valid):
                break
        # Start adding items starting from highest utility if feasible
        for i in self.utility.flip(-1):
            changeable = s[:,i] == 0
            s[changeable, i] = 1
            valid = self.is_valid(s)
            s[~valid & changeable, i] = 0
        return (s*2) - 1
