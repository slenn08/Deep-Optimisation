from typing import Tuple
import torch

from .Models.DOBase import DOBase
from .COProblems import OptimizationProblem
from .Data.Functions import to_int_list

def optimize_solutions(solutions : torch.Tensor, fitnesses : torch.Tensor,
                            model : DOBase, problem : OptimizationProblem,
                            change_tolerance : int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    evaluations = 0
    last_improve = torch.zeros_like(fitnesses)

    while True:
        new_solutions = model.vary(solutions)

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
                if new_fitness == problem.max_fitness:
                    return (solutions, fitnesses, evaluations)  
            else:
                last_improve[i] += 1

        if torch.all(last_improve > change_tolerance):
            return (solutions, fitnesses, evaluations)            
