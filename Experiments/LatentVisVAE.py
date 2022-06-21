import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import itertools
import sys

sys.path.append(".")
from Models.DOVAE2 import *
from data import data, linkages

# Hyper-parameters
change_tolerance = 512
batch_size = 16
window_size = 32
lr = 0.001
kl_weight = 0.1
compression = 0.8
pop_size = 128

for c, e, problem_size in itertools.product(["nov","ov","ndov","npov"],["gc","hgc","rs"],[16,32,64,128,256]):
    print(c,e,problem_size)
    problem_string = "{}_{}_{}".format(c,e,problem_size)
    linkage = None
    if e == "hgc":
        linkage = linkages[problem_string]
    problem = ECProblem(problem_size, c, e, linkage)
    population = data[problem_string][:pop_size]
    problem.max_fitness = float('inf')
    latent_size = int(compression*problem_size)

    model = DeepOptimizer(problem_size, latent_size)
    model.to(device)

    print("Learning")
    population = list(map(lambda x : (torch.tensor(x[0], dtype=torch.float32, device=device), x[1]), population))
    model.reset_weights()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    learn_from_population(model, population, optimizer, batch_size, latent_size, kl_weight)

    # Sort based on fitness
    population.sort(key=lambda x:x[1], reverse=True)
    n = 100
    fitnesses_grid = np.zeros((len(population)-1,n))
    s1,f1 = population[0]

    for i,(s2,f2) in enumerate(population[1:]):

        with torch.no_grad():
            latent1,_ = model.encode(s1)
            latent2,_ = model.encode(s2)
            points = np.linspace(latent1, latent2, n)
            fitnesses = []
            for point in points[:int(len(points)/2)]:
                new_reconstruction = torch.sign(model.decode(torch.tensor(point, dtype=torch.float32)))
                delta_s = new_reconstruction - s1
                new_solution = s1 + delta_s
                new_solution = to_int_list(new_solution)
                # new_solution = to_int_list(new_reconstruction)
                fitnesses.append(problem.fitness(new_solution))

            for point in points[int(len(points)/2):]:
                new_reconstruction = torch.sign(model.decode(torch.tensor(point, dtype=torch.float32)))
                delta_s = new_reconstruction - s2
                new_solution = s2 + delta_s
                new_solution = to_int_list(new_solution)
                # new_solution = to_int_list(new_reconstruction)
                fitnesses.append(problem.fitness(new_solution))
            
            fitnesses_grid[i] = fitnesses

    mean_fitnesses = np.mean(fitnesses_grid, axis=0)

    fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(4,4))

    img1 = ax1.imshow(fitnesses_grid, cmap='hot', interpolation='none')
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(img1, cax=cax1)
    cbar.ax.set_ylabel('Fitness', rotation=270, linespacing=2.0)
    ax1.set_title("C={}, E={}, Size={}".format(c.upper(),e.upper(),problem_size),
                  fontdict={'fontsize': 8, 'fontweight': 'medium'})
    ax1.set_ylabel("Solutions Ranked By Fitness", fontdict={'fontsize': 8, 'fontweight': 'medium'})


    ax2.plot(list(range(len(mean_fitnesses))), mean_fitnesses)
    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    ax2.set_aspect(asp)
    ax2.set_title("Av. Fitness Over All Paths", fontdict={'fontsize': 8, 'fontweight': 'medium'})
    ax2.set_ylabel("Fitness", fontdict={'fontsize': 8, 'fontweight': 'medium'})

    fig.subplots_adjust(hspace=0.3)
    plt.xticks(fontsize=8)
    path = "Graphs\\LatentVisVAE\\{}\\{}".format(c.upper(),e.upper())
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path+"\\{}.png".format(problem_size), bbox_inches='tight', dpi=300)
    # plt.show()
