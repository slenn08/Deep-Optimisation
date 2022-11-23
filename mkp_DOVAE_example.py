import torch

from COProblems.MKP import MKP
from COProblems.QUBO import QUBO
from Models.DOVAE import DOVAE
from OptimVAE import OptimVAEHandler

# Highly recommended to keep as cpu for problems of size <= 100
device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"
print(device)
device = torch.device(device)

change_tolerance = 100
problem_size = 100
pop_size = 100
problem = MKP("COProblems\\mkp\\problems30d.txt", "COProblems\\mkp\\fitnesses30d.txt", 0, device=device)
#problem = QUBO("COProblems\\qubo\\bqp1000.txt", 0, device=device)
print("Max fitness: {}".format(problem.max_fitness))

lr = 0.002
compression_ratio = 0.8
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
    # Learing with the entire population as a batch is technically not the best from a machine learning perspective,
    # but does not seem to have a massive impact on solution quality whilst also increasing learning speed significantly.
    handler.learn_from_population(population, optimizer, batch_size=pop_size)
    print("learnt")
    population, fitnesses, evaluations, done = handler.optimise_solutions(
        population, fitnesses, change_tolerance, repair_solutions=True
    )
    handler.print_statistics(fitnesses)
    total_eval += evaluations
    print("Evaluations: {}".format(total_eval))

    # best_i = torch.argmax(fitnesses)
    # print("Best solution - fitness = {}".format(fitnesses[best_i].item()))
    # print(population[best_i].tolist())
    if done:
        break


