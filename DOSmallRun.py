import torch

from COProblems.OptimisationProblem import MKP
from Models.DOAE import DOAE
from Models.DOSmall import DOSmall
from OptimSmall import OptimSmallHandler

change_tolerance = 100
problem_size = 100
pop_size = 100
problem = MKP("COProblems\\mkp\\problems5d.txt", "COProblems\\mkp\\fitnesses5d.txt", 0)

dropout_prob = 0.0
l1_coef = 0
l2_coef = 0
lr = 0.002
compression_ratio = 0.8
model = DOSmall(problem_size)
handler = OptimSmallHandler(model, problem)

population, fitnesses = handler.generate_population(pop_size)
population, fitnesses, _, _ = handler.hillclimb(population, fitnesses, change_tolerance)

total_eval = 0
max_depth = 3
depth = 0
sizes = 15
while True:
    model.transition(sizes, points_number=problem_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    handler.learn_from_population(population, optimizer, print_loss=False)
    population, fitnesses, evaluations, done = handler.optimise_solutions(
        population, fitnesses, change_tolerance
    )
    handler.print_statistics(fitnesses)
    total_eval += evaluations
    print("Evaluations: {}".format(total_eval))

    best_i = torch.argmax(fitnesses)
    print("Best solution - fitness = {}".format(fitnesses[best_i].item()))
    print(population[best_i].tolist())
    if done:
        break


