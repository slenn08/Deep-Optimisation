import torch

from COProblems.OptimisationProblem import MKP
from Models.DOVAE import DOVAE
from OptimVAE import OptimVAEHandler

change_tolerance = 200
problem_size = 100
pop_size = 100
problem = MKP("COProblems\\mkp\\problems5d.txt", "COProblems\\mkp\\fitnesses5d.txt", 0)
print("Max fitness: {}".format(problem.max_fitness))

lr = 0.002
batch_size = 16
compression_ratio = 0.8
model = DOVAE(problem_size, round(compression_ratio * pop_size))
handler = OptimVAEHandler(model, problem)


population, fitnesses = handler.generate_population(pop_size)
population, fitnesses, _, _ = handler.hillclimb(population, fitnesses, change_tolerance)

total_eval = 0
depth = 0

while True:
    model.transition()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    handler.learn_from_population(population, optimizer)
    population, fitnesses, evaluations, done = handler.optimise_solutions(
        population, fitnesses, change_tolerance
    )
    handler.print_statistics(fitnesses)
    total_eval += evaluations
    print(total_eval)

    best_i = torch.argmax(fitnesses)
    print("Best solution - fitness = {}".format(fitnesses[best_i].item()))
    print(population[best_i].tolist())
    if done:
        break


