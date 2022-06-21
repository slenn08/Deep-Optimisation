import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import itertools
import sys

sys.path.append(".")
from Models.DO2 import *
from data import data, linkages

regs = [#nOV
        # 16, 32, 64, 128, 256
        (0.001,0.001),(0.001,0.001),(0.0005,0.0005),(0.0005,0.0005),(0.0002,0.0002), #gc
        (0.001,0.001),(0.001,0.001),(0.0005,0.0005),(0.0005,0.0005),(0.0002,0.0002), #hgc
        (0.001,0.001),(0.001,0.001),(0.0005,0.0005),(0.0002,0.0002),(0.0001,0.0001), #rs
        # OV
        (0.001,0.001),(0.001,0.001),(0.0005,0.0005),(0.0002,0.0002),(0.0001,0.00005),
        (0.001,0.001),(0.001,0.001),(0.0005,0.0005),(0.0005,0.0005),(0.0001,0.00005),
        (0.001,0.001),(0.001,0.001),(0.0005,0.0005),(0.0005,0.0005),(0.0001,0.00005),
        # nDOV
        (0.001,0.001),(0.001,0.001),(0.0005,0.0003),(0.0002,0.0002),(0.0001,0.00005),
        (0.001,0.001),(0.001,0.001),(0.0005,0.0003),(0.0002,0.0002),(0.0001,0.00005),
        (0.0005,0.0005),(0.0002,0.0002),(0.0002,0.0002),(0.0001,0.0001),(0.0001,0.00005),
        # nPOV
        (0.001,0.001),(0.0005,0.0005),(0.0002,0.0001),(0.0001,0.00005),(0.00005,0.000025),
        (0.001,0.001),(0.0005,0.0005),(0.0002,0.0001),(0.0001,0.00005),(0.00005,0.000025),
        (0.001,0.001),(0.0005,0.0005),(0.0002,0.0001),(0.0001,0.00005),(0.00005,0.000025)]

problems = itertools.product(["nov","ov","ndov","npov"],["gc","hgc","rs"],[16,32,64,128,256])
reg_dict = {"{}_{}_{}".format(c,e,problem_size) : r for (c,e,problem_size),r in zip(list(problems),regs)}
problems = itertools.product(["nov","ov","ndov","npov"],["gc","hgc","rs"],[16,32,64,128,256])
pop_size = 128

for c, e, problem_size in problems:
    if c != "nov" and e != "gc":
        continue
    print(c,e,problem_size)
    problem_string = "{}_{}_{}".format(c,e,problem_size)
    l1_reg, l2_reg = reg_dict[problem_string]
    linkage = None
    if e == "hgc":
        linkage = linkages[problem_string]
    problem = ECProblem(problem_size, c, e, linkage)
    population = data[problem_string][:pop_size]
    problem.max_fitness = float('inf')

    model = DeepOptimizer(problem_size)
    model.add_layer(int(compression*problem_size))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    criterion = nn.MSELoss()
    learn_from_population(model, population, criterion, optimizer, l1_reg)

    # Sort based on fitness
    population.sort(key=lambda x:x[1], reverse=True)
    n = 100
    fitnesses_grid = np.zeros((len(population)-1,n))
    s1,f1 = population[0]
    s1_tensor = torch.tensor(s1, dtype=torch.float32)

    for i,(s2,f2) in enumerate(population[1:]):
        with torch.no_grad():
            s2_tensor = torch.tensor(s2, dtype=torch.float32)
            latent1 = model.encode(s1_tensor, 1)
            latent2 = model.encode(s2_tensor, 1)

            points = np.linspace(latent1, latent2, 100)
            fitnesses = []
            for point in points[:int(len(points)/2)]:
                new_reconstruction = torch.sign(model.decode(torch.tensor(point, dtype=torch.float32), 1))
                delta_s = new_reconstruction - s1_tensor
                new_solution = s1_tensor + delta_s
                new_solution = to_int_list(new_solution)
                # new_solution = to_int_list(new_reconstruction)
                fitnesses.append(problem.fitness(new_solution))

            for point in points[int(len(points)/2):]:
                new_reconstruction = torch.sign(model.decode(torch.tensor(point, dtype=torch.float32), 1))
                delta_s = new_reconstruction - s2_tensor
                new_solution = s2_tensor + delta_s
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
    path = "Graphs\\LatentVisDO\\{}\\{}".format(c.upper(),e.upper())
    if not os.path.exists(path):
        os.makedirs(path)
    
    plt.savefig(path+"\\{}n.png".format(problem_size), bbox_inches='tight', dpi=300)
    # plt.show()

