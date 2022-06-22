import torch
from torch.utils.data import DataLoader

from typing import Tuple
from .Models.DOBase import DOBase
from .COProblems import OptimizationProblem
from .Data.Functions import to_int_list
from .Data.PopulationDataset import PopulationDataset

# Incomplete, need to make it general for hillclimb and MIV
def assess_changes(solutions : torch.Tensor, fitnesses : torch.Tensor,
                   new_solutions : torch.Tensor, problem : OptimizationProblem, 
                   change_tolerance : int, last_improve : torch.Tensor, 
                   evaluations : int) -> int:
    for i, (solution, new_solution, fitness) in enumerate(zip(solutions, new_solutions, fitnesses)):
        if torch.equal(solution, new_solution) or last_improve[i] > change_tolerance:
            last_improve[i] += 1
            continue
        #new_solution = problem.repair(new_solution)
        new_fitness = problem.fitness(to_int_list(new_solution))
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

def optimize_solutions(solutions : torch.Tensor, fitnesses : torch.Tensor,
                       model : DOBase, problem : OptimizationProblem,
                       change_tolerance : int) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
    evaluations = 0
    last_improve = torch.zeros_like(fitnesses)

    while True:
        new_solutions = model.vary(solutions)
        evaluations = assess_changes(solutions, fitnesses, new_solutions, problem, change_tolerance,
                                     last_improve, evaluations)
        if torch.any(fitnesses == problem.max_fitness): 
            return (solutions, fitnesses, evaluations, True)
        if torch.all(last_improve > change_tolerance):
            return (solutions, fitnesses, evaluations, False)   
        
                   
def hillclimb(solutions : torch.Tensor, fitnesses : torch.Tensor,
              change_tolerance : int, problem : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
    last_improve = torch.zeros_like(fitnesses)
    while True:
        new_solutions = solutions.clone().detach()
        i = torch.randint(0,new_solutions.shape[1], (new_solutions.shape[0],))
        # Provides values of either 1 or -1
        new_activations = torch.randint(0,2,i.shape,dtype=torch.float32) * 2 - 1
        new_solutions[torch.arange(new_solutions.shape[0]),i] = new_activations

        evaluations = assess_changes(solutions, fitnesses, new_solutions, problem, change_tolerance,
                                     last_improve, evaluations)
        if torch.any(fitnesses == problem.max_fitness): 
            return (solutions, fitnesses, evaluations, True)
        if torch.all(last_improve > change_tolerance):
            return (solutions, fitnesses, evaluations, False)   
        

def learn_from_population(model : DOBase, solutions : torch.Tensor,
                          optimizer : torch.optim.Optimizer, batch_size : int,
                          beta : float=0, l1_reg : int=0, model_type : str="AE") -> None:
    if model_type == "VAE":
        learn = lambda s : model.learn_from_sample(s, optimizer, beta)
    elif model_type == "AE":
        learn = lambda s : model.learn_from_sample(s, optimizer, l1_reg)

    epochs = 400
    for epoch in range(epochs):
        dataset = DataLoader(PopulationDataset(solutions), batch_size=batch_size, shuffle=True)
        for i,x in enumerate(dataset):
            loss = learn(x["solution"])
            # print("Epoch {}/{} - {}/{} - Loss = {}".format(epoch+1,epochs,i,len(population),loss))
    # show_mu_sd(model, x["solution"])