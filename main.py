import torch

from COProblems.OptimisationProblem import MKP, ECProblem, OptimisationProblem
from Models.DOAE import DOAE
from Models.DOVAE import DOVAE
from OptimAE import OptimAEHandler
from OptimVAE import OptimVAEHandler

change_tolerance = 128
problem_size = 100
compression = "ov"
environment = "gc"
pop_size = 100
problem = ECProblem(problem_size,compression,environment)
problem = MKP("COProblems\\mkp\\problems5d.txt", 1)

dropout_prob = 0.2
l1_coef = 0.0001
l2_coef = 0.0001
lr = 0.002
batch_size = 16
compression_ratio = 0.8
model = DOAE(problem_size, dropout_prob)
hidden_sizes = [int(problem_size*(compression_ratio**i)) for i in range(1,20)]

# lr = 0.002
# batch_size = 16
# compression_ratio = 0.8
# model = DOVAE(problem_size, round(compression_ratio*problem_size))

ae_handler = OptimAEHandler(model, problem)
#vae_handler = OptimVAEHandler(model, problem)


population, fitnesses = ae_handler.generate_population(pop_size)
print(torch.max(fitnesses))
population, fitnesses, _, _ = ae_handler.hillclimb(population, fitnesses, change_tolerance)
print(torch.max(fitnesses))


total_eval = 0
max_depth = 6
depth = 0
for hidden_size in hidden_sizes:
    if depth < max_depth:
        model.transition(hidden_size)
        depth += 1
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    ae_handler.learn_from_population(population, optimizer, batch_size, l1_coef)
    with torch.no_grad():
        population, fitnesses, evaluations, done = ae_handler.optimise_solutions(
            population, fitnesses, change_tolerance
        )
        print(torch.max(fitnesses))
    total_eval += evaluations
    print(total_eval)
    if done:
        break

# total_eval = 0
# done = False
# while not done:
#     model.transition()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     vae_handler.learn_from_population(population, optimizer, batch_size, 0.1)
#     population, fitnesses, evaluations, done = vae_handler.optimise_solutions(
#         population, fitnesses, change_tolerance
#     )
#     print(torch.max(fitnesses))
#     total_eval += evaluations
#     print(total_eval)


