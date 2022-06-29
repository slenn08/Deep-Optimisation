import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import os
import sys

sys.path.append(".")
from Models.DOVAE2 import *
from data import data, linkages

# Hyper-parameters
change_tolerance = 512
pop_size = 128
batch_size = 16
window_size = 32
lr = 0.001
kl_weight = 0.1
compression = 0.8
no_change_prob = 0.9

for c, e, problem_size in itertools.product(["nov","ov","ndov","npov"],["gc","hgc","rs"],[16,32,64,128,256]):
    print(c,e,problem_size)
    latent_size = int(compression*problem_size)
    problem_string = "{}_{}_{}".format(c,e,problem_size)
    linkage = None
    if e == "hgc":
        linkage = linkages[problem_string]
    problem = ECProblem(problem_size, c, e, linkage)
    population = data[problem_string][:pop_size]
    model = DeepOptimizer(problem_size, int(compression*problem_size))

    model.reset_weights()
    population = list(map(lambda x : (torch.tensor(x[0], dtype=torch.float32, device=device), x[1]), population))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    learn_from_population(model, population, optimizer, batch_size, latent_size, kl_weight)

    population.sort(key=lambda x:x[1], reverse=True)
    with torch.no_grad():
        for solution, fitness in [population[-1]]:
            hidden_repr, logvar = model.encode(torch.tensor(solution, dtype=torch.float32))
            std = torch.exp(0.5*logvar)

            old_reconstruction = torch.sign(model.decode(hidden_repr))
            # print(problem.fitness(to_int_list(old_reconstruction)))

            negative_change = 0
            no_change = 0
            positive_change = 0

            min_z = hidden_repr - 20*std
            max_z = hidden_repr + 20*std
            n = 100
            fitness_changes = np.zeros((hidden_repr.shape[0],n))
            # For each variable
            for i, (var_min, var_max) in enumerate(zip(min_z, max_z)):
                new_hidden_repr = hidden_repr.clone()
                fitnesses = []
                # For each new value in the variable
                for j,new_var in enumerate(np.linspace(var_min, var_max, n)):
                    new_hidden_repr[i] = new_var
                    new_reconstruction = torch.sign(model.decode(new_hidden_repr))
                    delta_s = new_reconstruction - old_reconstruction
                    new_solution = torch.tensor(solution, dtype=torch.float32) + delta_s
                    new_solution = to_int_list(new_solution)
                    new_fitness = problem.fitness(new_solution)

                    d_fitness = new_fitness - fitness
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
            ax1.set_title("C={}, E={}, Size={}".format(c.upper(),e.upper(),problem_size),
                          fontdict={'fontsize': 12, 'fontweight': 'medium'})
            ax1.set_ylabel("Variable", fontdict={'fontsize': 12, 'fontweight': 'medium'})
            ax1.set_xlabel("S.D Away From the Mean", fontdict={'fontsize': 12, 'fontweight': 'medium'})
            # plt.imshow(fitness_changes, cmap='hot', interpolation='none', extent=[-20,20,0,len(hidden_repr)])
            # plt.colorbar()
            # plt.show()
            path = "Graphs\\MutantSpectraVAELast\\{}\\{}".format(c.upper(),e.upper())
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path+"\\{}.png".format(problem_size), bbox_inches='tight', dpi=300)
            plt.close()
            #plt.savefig("test.png", bbox_inches='tight', dpi=300)

        # print(negative_change)
        # print(no_change)
        # print(positive_change)


    # hidden_repr_0, logvar_0 = model.encode(torch.tensor(population[0][0], dtype=torch.float32))
    # old_reconstruction_0 = torch.sign(model.decode(hidden_repr_0))
    # for solution, fitness in population[1:]:
    #     hidden_repr, logvar = model.encode(torch.tensor(solution, dtype=torch.float32))
    #     old_reconstruction = torch.sign(model.decode(hidden_repr))

    #     new_hidden_repr = (hidden_repr + hidden_repr_0) / 2

    #     new_reconstruction = torch.sign(model.decode(new_hidden_repr))
    #     delta_s = (new_reconstruction - old_reconstruction + new_reconstruction - old_reconstruction_0)
    #     new_solution = torch.tensor(solution, dtype=torch.float32) + torch.tensor(population[0][0], dtype=torch.float32) + delta_s
    #     new_solution = to_int_list(new_solution)
    #     new_fitness = problem.fitness(new_solution)
    #     print("{} & {} = {}".format(population[0][1], fitness, new_fitness))
