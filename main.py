import torch

from Optimise import hillclimb
from COProblems.OptimizationProblem import ECProblem, OptimizationProblem
from Models.DO import DO
from Models.DOVAE import DOVAE
from OptimHandler import OptimHandler
from OptimAE import OptimAEHandler
from OptimVAE import OptimVAEHandler
from Data.PopulationDataset import PopulationDataset
from Data.Functions import generate_population

change_tolerance = 256
problem_size = 128
compression = "ov"
environment = "rs"
pop_size = 128
problem = ECProblem(problem_size,compression,environment)

# dropout_prob = 0.2
# l1_coef = 0.0001
# l2_coef = 0.00005
# lr = 0.002
# batch_size = 16
# compression_ratio = 0.9
# model = DO(problem_size, dropout_prob)
# hidden_sizes = [int(problem_size*(compression_ratio**i)) for i in range(1,20)]

lr = 0.002
batch_size = 16
compression_ratio = 0.8
model = DOVAE(problem_size, round(compression_ratio*problem_size))


population, fitnesses = generate_population(problem, pop_size)
population, fitnesses, _, _ = hillclimb(population, fitnesses, change_tolerance, problem)
print(torch.max(fitnesses))

# ae_handler = OptimAEHandler(model, problem)
vae_handler = OptimVAEHandler(model, problem)

total_eval = 0
# for hidden_size in hidden_sizes:
#     model.transition(hidden_size)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
#     ae_handler.learn_from_population(population, optimizer, batch_size, l1_coef)
#     with torch.no_grad():
#         population, fitnesses, evaluations, done = ae_handler.optimise_solutions(
#             population, fitnesses, change_tolerance
#         )
#         print(torch.max(fitnesses))
#     total_eval += evaluations
#     print(total_eval)
#     if done:
#         break

total_eval = 0
done = False
while not done:
    model.transition()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    vae_handler.learn_from_population(population, optimizer, batch_size, 0.1)
    population, fitnesses, evaluations, done = vae_handler.optimise_solutions(
        population, fitnesses, change_tolerance
    )
    print(torch.max(fitnesses))
    total_eval += evaluations
    print(total_eval)


