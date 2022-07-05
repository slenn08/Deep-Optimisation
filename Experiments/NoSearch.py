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

change_tolerance = 64
problem_size = 256
compression = "npov"
environment = "hgc"
pop_size = 128
problem_string = "{}_{}_{}".format(compression,environment,problem_size)
linkage = None
if environment == "hgc":
    linkage = linkages[problem_string]
problem = ECProblem(problem_size,compression,environment,linkages=linkage)

lr = 0.001
batch_size = 16
compression_ratio = 0.8
model = DOVAE(problem_size, round(compression_ratio*problem_size))
vae_handler = OptimVAEHandler(model, problem)

# population, fitnesses = generate_population(problem, pop_size)
# population, fitnesses, _, _ = hillclimb(population, fitnesses, change_tolerance, problem)
# print_statistics(fitnesses)

population = data[problem_string][:pop_size]
fitnesses = torch.tensor(list(map(lambda x : x[1], population)), dtype=torch.float32)
population = torch.tensor(list(map(lambda x : x[0], population)), dtype=torch.float32)
print_statistics(fitnesses)

model.transition()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
vae_handler.learn_from_population(population, optimizer, batch_size, 0.1)

print(population.shape)
print(torch.unique(population, dim=0).shape)
print(torch.unique(torch.sign(model(population)[0]), dim=0).shape)


with torch.no_grad():
    # Sort population and fitnesses by fitness ascending
    fitnesses, i = torch.sort(fitnesses)
    population = population[i]
    fitness_changes = []
    
    used_indices = 32
    i = torch.arange(0,pop_size)
    #i = torch.randint(0, population.shape[0], (used_indices,))
    population_hm = population.numpy()[i]
    new_population = torch.sign(model(population)[0])
    new_population_hm = new_population.numpy()[i]
    changes_hm = (new_population_hm - population_hm) / 2 
    self_h_distance = np.abs(changes_hm).sum(axis=1)

    for i, (s, f) in enumerate(zip(population[i], fitnesses[i])):
        hidden_repr, logvar = model.encode(s)
        new_solution = torch.sign(model.decode(hidden_repr))
        new_solution = to_int_list(new_solution)
        new_fitness = problem.fitness(new_solution)
        d_fitness = new_fitness - f
        fitness_changes.append(d_fitness)
    
    indices = [i for i in range(len(fitness_changes))]
    #plt.plot(indices, fitness_changes)
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(ncols=5,figsize=(20,4))
    img1 = ax1.imshow(population_hm, cmap='hot', interpolation='none')
    img2 = ax2.imshow(new_population_hm, cmap='hot', interpolation='none')
    img3 = ax3.imshow(changes_hm, cmap='hot', interpolation='none')
    ax4.barh(indices, fitness_changes)
    ax5.barh(indices, self_h_distance)
    plt.show()


# new_fitnesses = torch.zeros_like(fitnesses)
# for i,s in enumerate(new_reconstruction):
#     new_fitnesses[i] = problem.fitness(to_int_list(s))
# print_statistics(new_fitnesses)
# print(new_fitnesses - fitnesses)