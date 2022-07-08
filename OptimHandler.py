import torch
from typing import Tuple
from abc import abstractmethod, ABC

from COProblems.OptimisationProblem import OptimisationProblem
from Models.DOBase import DOBase

class OptimHandler(ABC):
    """
    Abstract class to handle the DO algorithm, given a model and a problem. Each handler
    is specific to the model as different model types may require slightly different algorithms.
    """
    def __init__(self, model : DOBase, problem : OptimisationProblem):
        """
        Constructor method for the OptimHandler class.

        Args:
            model: DOBase
                The central model being used for DO.
            problem: OptimisationProblem
                The combinatorial optimisation problem being solved.
        """
        self.model = model
        self.problem = problem
    
    @abstractmethod
    def learn_from_population(self, solutions: torch.Tensor,
                              optimizer: torch.optim.Optimizer, batch_size: int,
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
    def optimise_solutions(self, solutions: torch.Tensor, fitnesses: torch.Tensor, change_tolerance: int,
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

    # Incomplete, need to make it general for hillclimb and MIV
    @torch.no_grad()
    def assess_changes(self, solutions: torch.Tensor, fitnesses: torch.Tensor,
                       new_solutions: torch.Tensor, change_tolerance: int,
                       last_improve: torch.Tensor) -> int:
        """
        Determines which changes to solutions are positive and neutral and should be kept, and 
        which changes are negative and should be discarded. Solutions and fitnesses are modified
        in-place during this process.

        Args:
            solutions: torch.Tensor
                The current solutions. Has shape N x W, where N is the number of solutions
                in the population and W is the size of each solution.
            fitnesses: torch.Tensor
                The list of fitnesses relating to each solution. Has shape N, where the ith fitness
                corresponds to the ith solution in the solutions tensor.
            new_solutions: torch.Tensor
                The new proposed solutions after Model-Informed Variation has been applied.
            change_tolerance: int
                Defines how many neutral or negative fitness changes can be made in a row before a 
                solution is deemed an optima during the optimisation process.
            last_improve: torch.Tensor
                A list of numbers containing how many attempts have been made at making a change to 
                each solution without encountering a positive solution. If the ith element is greater
                than change_tolerance, no more changes shall be made to the ith solution. Has shape N.
        
        Returns:
            The number of evaluations that have been made during this function call.
        """
        evaluations = 0
        for i, (solution, new_solution, fitness) in enumerate(zip(solutions, new_solutions, fitnesses)):
            if torch.equal(solution, new_solution) or last_improve[i] > change_tolerance:
                last_improve[i] += 1
                continue
            new_fitness = self.problem.fitness(self.to_int_list(new_solution))
            evaluations += 1

            if new_fitness >= fitness:
                if new_fitness > fitness:
                    last_improve[i] = 0 
                else:
                    last_improve[i] += 1
                solutions[i] = new_solution
                fitnesses[i] = new_fitness       
            else:
                last_improve[i] += 1

        return evaluations  
        
    @torch.no_grad() 
    def hillclimb(self, solutions: torch.Tensor, fitnesses: torch.Tensor,
                  change_tolerance: int) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
        """
        Locally optimises solutions using a bit-substitution hill climber.

        Args
            solutions: torch.Tensor
                The current solutions. Has shape N x W, where N is the number of solutions
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
        last_improve = torch.zeros_like(fitnesses)
        while True:
            new_solutions = solutions.clone().detach()
            # Select which bits to flip
            i = torch.randint(0,new_solutions.shape[1], (new_solutions.shape[0],))
            # Flips the selected bits
            new_solutions[torch.arange(new_solutions.shape[0]),i] *= -1

            _ = self.assess_changes(solutions, fitnesses, new_solutions, change_tolerance,
                                    last_improve, 0)
            if torch.any(fitnesses == self.problem.max_fitness): 
                return (solutions, fitnesses, 0, True)
            if torch.all(last_improve > change_tolerance):
                return (solutions, fitnesses, 0, False)
    
    def to_int_list(self, x):
        """
        Converts a tensor of floats into a list of 1s and -1s
        """
        try:
            x = torch.sign(x)
            x = x.tolist()
            x = [int(i) for i in x]  
        except TypeError:
            pass
        return x

    def generate_population(self, pop_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a random solution to the given problem.

        Args:
            pop_size: int
                The number of solutions in the population.
        
        Returns:
            A tuple of tensors containing the randomly generated solutions and their associated
            fitnesses.
        """
        population = [self.problem.random_solution() for _ in range(pop_size)]
        fitnesses = [self.problem.fitness(x) for x in population]
        population = torch.tensor(population, dtype=torch.float32)
        fitnesses = torch.tensor(fitnesses, dtype=torch.float32)
        return population, fitnesses

    def print_statistics(self, fitnesses: torch.tensor):
        """
        Prints basic statistics about the fitnesses of a population.

        Args:
            fitnesses: torch.Tensor
                The fitnesses of the population.
        """
        mean_f = torch.mean(fitnesses).item()  
        max_f = torch.max(fitnesses).item()

        print("Max : {}, Mean : {}".format(max_f, mean_f))     