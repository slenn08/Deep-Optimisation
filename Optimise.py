import torch
from torch.utils.data import DataLoader

from typing import Tuple
from .COProblems.OptimizationProblem import OptimizationProblem
from .Data.Functions import to_int_list

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