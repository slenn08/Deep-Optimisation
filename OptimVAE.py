import torch
from torch.utils.data import DataLoader
from typing import Tuple

from .Optimise import assess_changes
from .COProblems.OptimizationProblem import OptimizationProblem
from .Models.DOVAE import DOVAE
from .OptimHandler import OptimHandler
from .Data import PopulationDataset

class OptimVAEHandler(OptimHandler):
    def __init__(self, model: DOVAE, problem: OptimizationProblem):
        super().__init__(model, problem)
    
    def learn_from_population(self, solutions: torch.Tensor, optimizer: torch.optim.Optimizer,
                              batch_size: int, beta : float, epochs : int=400) -> None:
        for epoch in range(epochs):
            dataset = DataLoader(PopulationDataset(solutions), batch_size=batch_size, shuffle=True)
            for i,x in enumerate(dataset):
                loss = self.model.learn_from_sample(x["solution"], optimizer, beta)
                # print("Epoch {}/{} - {}/{} - Loss = {}".format(epoch+1,epochs,i,len(population),loss))
        # show_mu_sd(model, x["solution"])

    def optimise_solutions(self, solutions: torch.Tensor, fitnesses: torch.Tensor,
                           change_tolerance : int) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
        evaluations = 0
        last_improve = torch.zeros_like(fitnesses)

        while True:
            new_solutions = self.model.vary(solutions)
            evaluations = assess_changes(solutions, fitnesses, new_solutions, self.problem, change_tolerance,
                                        last_improve, evaluations)
            if torch.any(fitnesses == self.problem.max_fitness): 
                return (solutions, fitnesses, evaluations, True)
            if torch.all(last_improve > change_tolerance):
                return (solutions, fitnesses, evaluations, False)   
        
