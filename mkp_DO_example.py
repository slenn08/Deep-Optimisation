import torch

from COProblems.MKP import MKP
from COProblems.QUBO import QUBO
from Models.DOAE import DOAE
from OptimAE import OptimAEHandler

# Highly recommended to keep as cpu for problems of size <= 100
device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"
print(device)
device = torch.device(device)

change_tolerance = 100
problem_size = 100
pop_size = 100
problem = MKP("COProblems\\mkp\\problems30d.txt", "COProblems\\mkp\\fitnesses30d.txt", 12, device)
#problem = QUBO("COProblems\\qubo\\bqp500.txt", 0, device)

dropout_prob = 0.2
# 1000 bit QUBO l1 and l2:
# l1_coef = 0.0000025
# l2_coef = 0.0000025
# 500 bit QUBO l1 and l2:
# l1_coef = 0.000005
# l2_coef = 0.000005
# 100 bit l1 and l2
l1_coef = 0.0001
l2_coef = 0.0001
lr = 0.002
compression_ratio = 0.8
model = DOAE(problem_size, dropout_prob, device)
hidden_size = problem_size
handler = OptimAEHandler(model, problem, device)

population, fitnesses = handler.generate_population(pop_size)
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
    print("Learning from population")
    # Learing with the entire population as a batch is technically not the best from a machine learning perspective,
    # but does not seem to have a massive impact on solution quality whilst also increasing learning speed significantly.
    handler.learn_from_population(population, optimizer, l1_coef=l1_coef, batch_size=pop_size)
    print("Optimising population")
    population, fitnesses, evaluations, done = handler.optimise_solutions(
        population, fitnesses, change_tolerance, encode=True, repair_solutions=True, deepest_only=False
    )
    handler.print_statistics(fitnesses)
    total_eval += evaluations
    print("Evaluations: {}".format(total_eval))
    
    # Uncomment lines below to print out best solution at every transition
    # best_i = torch.argmax(fitnesses)
    # print("Best solution - fitness = {}".format(fitnesses[best_i].item()))
    # print(population[best_i].tolist())
    if done:
        break


