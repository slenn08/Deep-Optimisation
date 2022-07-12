import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

from COProblems.OptimisationProblem import OptimisationProblem
from Models.DOVAE import DOVAE
from OptimHandler import OptimHandler

class OptimVAEHandler(OptimHandler):
    """
    Describes the algorithm for carrying out DO with a VAE model.
    """
    def __init__(self, model: DOVAE, problem: OptimisationProblem):
        """
        Constructor method for OptimVAEHandler.

        Args:
            model: DO
                The central VAE model used in Deep Optimisation.
            problem: OptimisationProblem
                The problem being solved.
        """
        super().__init__(model, problem)
    
    def learn_from_population(self, solutions: torch.Tensor, optimizer: torch.optim.Optimizer,
                              batch_size: int, beta: float, epochs: int=400) -> None:
        """
        Method to make the VAE learn from the population of solutions.

        Args:
            solutions: torch.Tensor
                The solutions to learn from. Has shape N x W, where N is the number of solutions
                in the population and W is the size of each solution.
            optimizer: torch.optim.Optimizer
                The optimizer used to adjust the weights of the model.
            batch_size: int
                The batch size used during the learning process.
            BETA: int
                The coefficient of the KL Divergence term in the loss function.
            epochs: int
                The number of epochs to train for.
        """
        for epoch in range(epochs):
            dataset = DataLoader(TensorDataset(solutions), batch_size=batch_size, shuffle=True)
            for i,x in enumerate(dataset):
                loss = self.model.learn_from_sample(x[0], optimizer, beta)
                # print("Epoch {}/{} - {}/{} - Loss = {}".format(epoch+1,epochs,i,len(solutions),loss["recon"].item()))
        # show_mu_sd(model, x["solution"])
    
    @torch.no_grad()
    def optimise_solutions(self, solutions: torch.Tensor, fitnesses: torch.Tensor,
                           change_tolerance : int) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
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
        evaluations = 0
        last_improve = torch.zeros_like(fitnesses)

        while True:
            new_solutions = self.model.vary(solutions)
            evaluations += self.assess_changes(solutions, fitnesses, new_solutions, change_tolerance,
                                              last_improve)
            if torch.any(fitnesses == self.problem.max_fitness): 
                return (solutions, fitnesses, evaluations, True)
            if torch.all(last_improve > change_tolerance):
                return (solutions, fitnesses, evaluations, False)   
        
