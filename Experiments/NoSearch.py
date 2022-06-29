import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.append(".")
from Optimise import hillclimb
from COProblems.OptimizationProblem import ECProblem
from Models.DOVAE import DOVAE
from OptimVAE import OptimVAEHandler
from Data.Functions import generate_population, to_int_list, print_statistics
from data import data, linkages

change_tolerance = 256
problem_size = 256
compression = "nov"
environment = "hgc"
pop_size = 128
problem = ECProblem(problem_size,compression,environment)

lr = 0.001
batch_size = 16
compression_ratio = 0.8
model = DOVAE(problem_size, round(compression_ratio*problem_size))
vae_handler = OptimVAEHandler(model, problem)

# population, fitnesses = generate_population(problem, pop_size)
# population, fitnesses, _, _ = hillclimb(population, fitnesses, change_tolerance, problem)
# print_statistics(fitnesses)

population = data["nov_hgc_256"][:pop_size]
fitnesses = torch.tensor(list(map(lambda x : x[1], population)), dtype=torch.float32)
population = torch.tensor(list(map(lambda x : x[0], population)), dtype=torch.float32)
print_statistics(fitnesses)

model.transition()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
vae_handler.learn_from_population(population, optimizer, batch_size, 0.1)

with torch.no_grad():
    i = torch.argmin(fitnesses)
    s = population[0]
    print(fitnesses[i])
    # s = population[0]
    # print(fitnesses[0])
    hidden_repr, logvar = model.encode(s)
    std = torch.exp(0.5 * logvar)

    old_reconstruction = torch.sign(model.decode(hidden_repr))

    min_z = hidden_repr - 20*std
    max_z = hidden_repr + 20*std
    n = 100
    fitness_changes = np.zeros((hidden_repr.shape[0],n))
    # For each variable
    for i, (var_min, var_max) in enumerate(zip(min_z, max_z)):
        new_hidden_repr = hidden_repr.clone()
        # For each new value in the variable
        for j,new_var in enumerate(np.linspace(var_min, var_max, n)):
            new_hidden_repr[i] = new_var
            new_reconstruction = torch.sign(model.decode(new_hidden_repr))
            delta_s = new_reconstruction - old_reconstruction
            new_solution = s + delta_s
            new_solution = to_int_list(new_solution)
            new_fitness = problem.fitness(new_solution)

            d_fitness = new_fitness - fitnesses[0]
            # negative_d_fitness = 1/n if d_fitness < 0 else 0
            # zero_d_fitness = 1/n if d_fitness == 0 else 0
            # positive_d_fitness = 1/n if d_fitness > 0 else 0

            # negative_change += negative_d_fitness / len(min_z)
            # no_change += zero_d_fitness / len(min_z)
            # positive_change += positive_d_fitness / len(min_z)

            # fitnesses.append(new_fitness - fitness)
            # 1 if positive, 0 if no change, -1 if negative
            # fitness_changes[i][j] = d_fitness/abs(d_fitness) if d_fitness != 0 else 0
            fitness_changes[i][j] = d_fitness
        
        # fitnesses.append(variance[i].item())
        # print(fitnesses)
    fig, ax1 = plt.subplots(nrows=1, figsize=(4,4))
    img1 = ax1.imshow(fitness_changes, cmap='hot', interpolation='none', extent=[-20,20,0,len(hidden_repr)])
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(img1, cax=cax1)
    cbar.ax.set_ylabel('Fitness Change', rotation=270, linespacing=2.0)
    ax1.set_aspect('auto')
    ax1.set_ylabel("Variable", fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax1.set_xlabel("S.D Away From the Mean", fontdict={'fontsize': 12, 'fontweight': 'medium'})
    plt.show()


# new_fitnesses = torch.zeros_like(fitnesses)
# for i,s in enumerate(new_reconstruction):
#     new_fitnesses[i] = problem.fitness(to_int_list(s))
# print_statistics(new_fitnesses)
# print(new_fitnesses - fitnesses)