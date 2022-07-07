import torch
from torch.utils.data import DataLoader
from typing import Tuple

from Optimise import assess_changes
from COProblems.OptimizationProblem import OptimizationProblem
from Models.DO import DO
from OptimHandler import OptimHandler
from Data.PopulationDataset import PopulationDataset

class OptimAEHandler(OptimHandler):
    """
    Describes the algorithm for carrying out DO with an AE model as specified in 
    "Deep Optimisation: Learning and Searching in Deep Representations of Combinatorial
    Optimisation Problems", Jamie Caldwell.
    """
    def __init__(self, model: DO, problem: OptimizationProblem):
        """
        Constructor method for OptimAEHandler.

        Args:
            model: DO
                The central AE model used in Deep Optimisation.
            problem: OptimizationProblem
                The problem being solved.
        """
        super().__init__(model, problem)
    
    def learn_from_population(self, solutions: torch.Tensor, optimizer: torch.optim.Optimizer,
                              batch_size: int, l1_coef: float, epochs: int=400) -> None:
        """
        Method to make the AE learn from the population of solutions.

        Args:
            solutions: torch.Tensor
                The solutions to learn from. Has shape N x W, where N is the number of solutions
                in the population and W is the size of each solution.
            optimizer: torch.optim.Optimizer
                The optimizer used to adjust the weights of the model.
            batch_size: int
                The batch size used during the learning process.
            l1_coef: int
                The coefficient of the L1 term in the loss function.
            epochs: int
                The number of epochs to train for.
        """
        for epoch in range(epochs):
            dataset = DataLoader(PopulationDataset(solutions), batch_size=batch_size, shuffle=True)
            for i,x in enumerate(dataset):
                loss = self.model.learn_from_sample(x["solution"], optimizer, l1_coef)
                # print("Epoch {}/{} - {}/{} - Loss = {}".format(
                #     epoch+1,epochs,(i+1)*batch_size,len(solutions),loss["recon"]
                # ))
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
        self.model.eval()
        evaluations = 0
        for layer in range(self.model.num_layers-1, 0, -1):
            print("Optimising from layer {}".format(layer))

            last_improve = torch.zeros_like(fitnesses)

            while True:
                new_solutions = self.model.vary(solutions, layer, False)
                evaluations = assess_changes(solutions, fitnesses, new_solutions, self.problem, change_tolerance,
                                            last_improve, evaluations)
                if torch.any(fitnesses == self.problem.max_fitness): 
                    return (solutions, fitnesses, evaluations, True)
                if torch.all(last_improve > change_tolerance):
                    break   

        return (solutions, fitnesses, evaluations, False)
