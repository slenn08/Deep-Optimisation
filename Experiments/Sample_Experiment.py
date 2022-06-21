import sys

sys.path.append(".")
from Models.DOVAE2 import *

no_change_prob = 0.9
batch_size = 16
window_size = 32
lr = 0.001
kl_weight = 0.1
compression = 0.8
change_tolerance = 64
pop_size = 64
problem_size = 64
c = "npov"
e = "gc"
problem = ECProblem(problem_size, c, e)
latent_size = int(compression*problem_size)

model = DeepOptimizer(problem_size, latent_size)
model.to(device)
population = [problem.random_solution() for _ in range(pop_size)]
population = list(map(lambda x : (torch.tensor(x, dtype=torch.float32, device=device), problem.fitness(x)), population))
population = optimize_population_hillclimb(population, problem, window_size, pop_size)

total_evaluations = 0

model.reset_weights()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
learn_from_population(model, population, optimizer, batch_size, latent_size, kl_weight)

while True:
    print("Optimising solutions")
    with torch.no_grad():
        new_population = sample(pop_size, latent_size, model, problem)
    population = new_population + population
    population = sorted(population, key=lambda x: x[1])
    population = population[len(population)//2:]
    population, total_evaluations, solved = optimize_population_model(population, model, problem,
                                                            total_evaluations,3.0, latent_size,
                                                            no_change_prob,change_tolerance,window_size,
                                                            pop_size)
    if solved:
        break
    
    # population_residue += population[:pop_size//4]
    # population_residue = population_residue[-pop_size:]

    model.reset_weights()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # learn_from_population(model, population+population_residue, optimizer, batch_size, latent_size, kl_weight)
    learn_from_population(model, population, optimizer, batch_size, latent_size, kl_weight)
