import numpy as np
import torch
from torch import nn
import itertools
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(".")
from Models import DO2 as DO
from Models import DOVAE2 as VDO
from COProblems.OptimizationProblem import *
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
# Should include 256 bit
problems = itertools.product(["nov","ov","ndov","npov"],["gc","hgc","rs"],[16,32,64,128,256])
reg_dict = {"{}_{}_{}".format(c,e,problem_size) : r for (c,e,problem_size),r in zip(list(problems),regs)}
problems = itertools.product(["nov","ov","ndov","npov"],["gc","hgc","rs"],[16,32,64,128,256])
# 
pop_size = 128
compression = 0.8
kl_weight = 0.1
batch_size = 16

for c, e, problem_size in problems:
    print(c,e,problem_size)
    latent_size = int(compression*problem_size)
    problem_string = "{}_{}_{}".format(c,e,problem_size)
    linkage = None
    if e == "hgc":
        linkage = linkages[problem_string]
    problem = ECProblem(problem_size, c, e, linkage)
    l1_reg, l2_reg = reg_dict[problem_string]
    population_DO = data[problem_string][:pop_size]
    population_VDO = list(map(lambda x : (torch.tensor(x[0], dtype=torch.float32), x[1]), population_DO))
    problem.max_fitness = float('inf')

    model_DO = DO.DeepOptimizer(problem_size)
    model_DO.add_layer(latent_size)

    model_VDO = VDO.DeepOptimizer(problem_size, latent_size)
    model_VDO.reset_weights()

    optimizer_DO = torch.optim.Adam(model_DO.parameters(), lr=0.002, weight_decay=l2_reg)
    optimizer_VDO = torch.optim.Adam(model_VDO.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    DO.learn_from_population(model_DO, population_DO, criterion, optimizer_DO, l1_reg)
    VDO.learn_from_population(model_VDO, population_VDO, optimizer_VDO, batch_size, latent_size, kl_weight)

    DO_change_dict = {}
    VDO5_change_dict = {}
    VDO10_change_dict = {}
    VDO15_change_dict = {}

    for solution, fitness in population_DO:
        with torch.no_grad():

            hidden_repr = model_DO.encode(torch.tensor(solution, dtype=torch.float32), 1)

            old_reconstruction = torch.sign(model_DO.decode(hidden_repr, 1))

            for i in range(len(hidden_repr)):
                new_hidden_repr = hidden_repr.clone()
                fitnesses = []
                for new_var in [1,-1]:
                    new_hidden_repr[i] = new_var
                    new_reconstruction = torch.sign(model_DO.decode(new_hidden_repr, 1))
                    delta_s = new_reconstruction - old_reconstruction
                    new_solution = torch.tensor(solution, dtype=torch.float32) + delta_s
                    new_solution = DO.to_int_list(new_solution)
                    new_fitness = problem.fitness(new_solution)

                    d_fitness = int(new_fitness - fitness)
                    DO_change_dict[d_fitness] = DO_change_dict.get(d_fitness, 0) + 1

    for solution, fitness in population_VDO:
        with torch.no_grad():

            hidden_repr, logvar = model_VDO.encode(solution)
            std = torch.exp(0.5*logvar)

            old_reconstruction = torch.sign(model_VDO.decode(hidden_repr))

            for i, (mean,stdi) in enumerate(zip(hidden_repr, std)):
                new_hidden_repr = hidden_repr.clone()
                for new_var in [mean - 5*stdi, mean + 5*stdi]:
                    new_hidden_repr[i] = new_var
                    new_reconstruction = torch.sign(model_VDO.decode(new_hidden_repr))
                    delta_s = new_reconstruction - old_reconstruction
                    new_solution = solution + delta_s
                    new_solution = VDO.to_int_list(new_solution)
                    new_fitness = problem.fitness(new_solution)

                    d_fitness = int(new_fitness - fitness)
                    VDO5_change_dict[d_fitness] = VDO5_change_dict.get(d_fitness, 0) + 1
                # new_hidden_repr = hidden_repr.clone()
                # for new_var in [mean - 10*stdi, mean + 10*stdi]:
                #     new_hidden_repr[i] = new_var
                #     new_reconstruction = torch.sign(model_VDO.decode(new_hidden_repr))
                #     delta_s = new_reconstruction - old_reconstruction
                #     new_solution = solution + delta_s
                #     new_solution = VDO.to_int_list(new_solution)
                #     new_fitness = problem.fitness(new_solution)

                #     d_fitness = int(new_fitness - fitness)
                #     VDO10_change_dict[d_fitness] = VDO10_change_dict.get(d_fitness, 0) + 1
                # new_hidden_repr = hidden_repr.clone()
                # for new_var in [mean - 15*stdi, mean + 15*stdi]:
                #     new_hidden_repr[i] = new_var
                #     new_reconstruction = torch.sign(model_VDO.decode(new_hidden_repr))
                #     delta_s = new_reconstruction - old_reconstruction
                #     new_solution = solution + delta_s
                #     new_solution = VDO.to_int_list(new_solution)
                #     new_fitness = problem.fitness(new_solution)

                #     d_fitness = int(new_fitness - fitness)
                #     VDO15_change_dict[d_fitness] = VDO15_change_dict.get(d_fitness, 0) + 1
                
    # total_samples = negative_change_DO + no_change_DO + positive_change_DO
    # negative_change_DO /= total_samples
    # no_change_DO /= total_samples
    # positive_change_DO /= total_samples

    # total_samples = negative_change_VDO5 + no_change_VDO5 + positive_change_VDO5
    # negative_change_VDO5 /= total_samples
    # no_change_VDO5 /= total_samples
    # positive_change_VDO5 /= total_samples

    # total_samples = negative_change_VDO10 + no_change_VDO10 + positive_change_VDO10
    # negative_change_VDO10 /= total_samples
    # no_change_VDO10 /= total_samples
    # positive_change_VDO10 /= total_samples

    # total_samples = negative_change_VDO15 + no_change_VDO15 + positive_change_VDO15
    # negative_change_VDO15 /= total_samples
    # no_change_VDO15 /= total_samples
    # positive_change_VDO15 /= total_samples

    # labels = ["Positive", "Zero", "Negative"]
    # DO_changes = [round(positive_change_DO*100,2), round(no_change_DO*100,2), round(negative_change_DO*100,2)]
    # VDO5_changes = [round(positive_change_VDO5*100,2), round(no_change_VDO5*100,2),
    #                round(negative_change_VDO5*100,2)]
    # VDO10_changes = [round(positive_change_VDO10*100,2), round(no_change_VDO10*100,2),
    #                round(negative_change_VDO10*100,2)]
    # VDO15_changes = [round(positive_change_VDO15*100,2), round(no_change_VDO15*100,2),
    #                round(negative_change_VDO15*100,2)]

    # x = np.arange(len(labels))  # the label locations
    # width = 0.2  # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x - 1.5*width, DO_changes, width, label='AE')
    # rects2 = ax.bar(x - 0.5*width, VDO5_changes, width, label='VAE (5x STD)')
    # rects3 = ax.bar(x + 0.5*width, VDO10_changes, width, label='VAE (10x STD)')
    # rects4 = ax.bar(x + 1.5*width, VDO15_changes, width, label='VAE (15x STD)')

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Percentage')
    # ax.set_title("C={}, E={}, Size={}".format(c.upper(),e.upper(),problem_size))
    # ax.set_xticks(x, labels)
    # ax.legend()
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    # ax.bar_label(rects3, padding=3)
    # ax.bar_label(rects4, padding=3)
    # ax.set_xlabel("Fitness Change Direction")

    # fig.tight_layout()
    
    # path = "Graphs\\MutantSpectraCompare\\{}\\{}".format(c.upper(),e.upper())
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # plt.savefig(path+"\\{}.png".format(problem_size), bbox_inches='tight', dpi=300)
    # plt.close()
    for change_dict, label in zip([DO_change_dict, VDO5_change_dict],
                                  ["AE", "VAE (5x STD)"]):
        fitnesses = list(change_dict.keys())
        min_change = min(fitnesses)
        max_change = max(fitnesses)
        fitness_changes = list(range(min_change, max_change+1))
        freq = []
        for fitness_change in fitness_changes:
            freq.append(change_dict.get(fitness_change, 0))
            #change_dict[fitness_change] = change_dict.get(fitness_change, 0)
        # freqs arent sorted...
        #freq = list(change_dict.values())
        total_freq = sum(freq)
        for i,f in enumerate(freq):
            freq[i] = f*100/total_freq
        plt.plot(fitness_changes, freq, label=label)
    plt.legend()
    plt.xlabel("Fitness Change")
    plt.ylabel("Percentage of Changes")
    plt.title("C={}, E={}, Size={}".format(c.upper(),e.upper(),problem_size))
    path = "Graphs\\MutantSpectraCompareContinous\\{}\\{}".format(c.upper(),e.upper())
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path+"\\{}.png".format(problem_size), bbox_inches='tight', dpi=300)
    plt.close()
    # plt.show()

