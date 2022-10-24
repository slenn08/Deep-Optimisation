import torch

from COProblems.OptimisationProblem import MKP, QUBO
from Models.DOVAE import DOVAE
from OptimVAE import OptimVAEHandler

change_tolerance = 500
problem_size = 1000
pop_size = 750
#problem = MKP("COProblems\\mkp\\problems5d.txt", "COProblems\\mkp\\fitnesses5d.txt", 0)
problem = QUBO("COProblems\\qubo\\bqp1000.txt", 0)
print("Max fitness: {}".format(problem.max_fitness))

lr = 0.002
batch_size = 750
compression_ratio = 0.8
device = "cuda" if torch.cuda.is_available() else "cpu"
#device="cpu"
print(device)
device = torch.device(device)
model = DOVAE(problem_size, round(compression_ratio * pop_size), device)
handler = OptimVAEHandler(model, problem, device)


population, fitnesses = handler.generate_population(pop_size)
handler.print_statistics(fitnesses)
population, fitnesses, _, _ = handler.hillclimb(population, fitnesses, change_tolerance)
handler.print_statistics(fitnesses)

total_eval = 0
depth = 0

while True:
    model.transition()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("learning")
    handler.learn_from_population(population, optimizer, batch_size=batch_size)
    print("learnt")
    population, fitnesses, evaluations, done = handler.optimise_solutions(
        population, fitnesses, change_tolerance
    )
    handler.print_statistics(fitnesses)
    total_eval += evaluations
    print("Evaluations: {}".format(total_eval))

    # best_i = torch.argmax(fitnesses)
    # print("Best solution - fitness = {}".format(fitnesses[best_i].item()))
    # print(population[best_i].tolist())
    if done:
        break


