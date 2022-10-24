import torch

from COProblems.OptimisationProblem import MKP, QUBO
from Models.DOAE import DOAE
from OptimAE import OptimAEHandler

change_tolerance = 1000
problem_size = 1000
pop_size = 750
#problem = MKP("COProblems\\mkp\\problems5d.txt", "COProblems\\mkp\\fitnesses5d.txt", 0)
problem = QUBO("COProblems\\qubo\\bqp1000.txt", 0)

dropout_prob = 0.2
l1_coef = 0.0000025
l2_coef = 0.0000025
lr = 0.002
compression_ratio = 0.8
device = "cuda" if torch.cuda.is_available() else "cpu"
#device="cpu"
print(device)
device = torch.device(device)
model = DOAE(problem_size, dropout_prob, device)
hidden_size = problem_size
handler = OptimAEHandler(model, problem, device)


population, fitnesses = handler.generate_population(pop_size)
handler.print_statistics(fitnesses)
population, fitnesses, _, _ = handler.hillclimb(population, fitnesses, change_tolerance)
handler.print_statistics(fitnesses)

total_eval = 0
max_depth = 6
depth = 0
while True:
    if depth < max_depth:
        hidden_size = round(hidden_size * compression_ratio)
        model.transition(hidden_size)
        depth += 1
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    print("learning")
    handler.learn_from_population(population, optimizer, l1_coef=l1_coef, batch_size=200)
    print("learnt")
    population, fitnesses, evaluations, done = handler.optimise_solutions(
        population, fitnesses, change_tolerance, encode=True
    )
    handler.print_statistics(fitnesses)
    total_eval += evaluations
    print("Evaluations: {}".format(total_eval))

    # best_i = torch.argmax(fitnesses)
    # print("Best solution - fitness = {}".format(fitnesses[best_i].item()))
    # print(population[best_i].tolist())
    if done:
        break


