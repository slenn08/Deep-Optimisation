from typing import Tuple
import torch

def to_int_list(x):
    try:
        x = torch.sign(x)
        x = x.tolist()
        x = [int(i) for i in x]  
    except TypeError:
        pass
    return x

def generate_population(problem, pop_size) -> Tuple[torch.Tensor, torch.Tensor]:
    population = [problem.random_solution() for _ in range(pop_size)]
    fitnesses = [problem.fitness(x) for x in population]
    population = torch.tensor(population, dtype=torch.float32)
    fitnesses = torch.tensor(fitnesses, dtype=torch.float32)
    return population, fitnesses
        