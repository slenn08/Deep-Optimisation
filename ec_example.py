import itertools

import torch
from matplotlib import pyplot as plt

from COProblems.ECProblem import ECProblem
from Models.DOAE import DOAE
from OptimAE import OptimAEHandler

"""
This script runs DO on the synthetic problems described by Jamie in his thesis. It solves
these problems, and outputs the number of evaluations taken to solve them on a graph. Different
problem types can be specified to ensure the performance of this implementation of DO is 
similar to that shown in Jamie's thesis.
"""

# Dictionary of all the regularisation coefficients used to solve the compression/environment
# problems
regs = [
        # (l1_coef, l2_coef)
        # nOV
        # 16, 32, 64, 128, 256
        (0.001,0.001),(0.001,0.001),(0.0005,0.0005),(0.0005,0.0005),(0.0002,0.0002), #gc
        (0.001,0.001),(0.001,0.001),(0.0005,0.0005),(0.0005,0.0005),(0.0002,0.0002), #hgc
        (0.001,0.001),(0.001,0.001),(0.0005,0.0005),(0.0002,0.0002),(0.0001,0.0001), #rs
        # OV
        (0.001,0.001),(0.001,0.001),(0.0005,0.0005),(0.0002,0.0002),(0.0001,0.00005),
        (0.001,0.001),(0.001,0.001),(0.0005,0.0005),(0.0002,0.0002),(0.0001,0.00005),
        (0.001,0.001),(0.001,0.001),(0.0005,0.0005),(0.0002,0.0002),(0.0001,0.00005),
        # nDOV
        (0.001,0.001),(0.001,0.001),(0.0005,0.0003),(0.0002,0.0002),(0.0001,0.00005),
        (0.001,0.001),(0.001,0.001),(0.0005,0.0003),(0.0002,0.0002),(0.0001,0.00005),
        (0.001,0.0005),(0.0005,0.0005),(0.00025,0.0005),(0.00005,0.00005),(0.0001,0.00001),
        # nPOV
        (0.001,0.001),(0.0005,0.0005),(0.0002,0.0001),(0.0001,0.00005),(0.00005,0.000025),
        (0.001,0.001),(0.0005,0.0005),(0.0002,0.0001),(0.0001,0.00005),(0.00005,0.000025),
        (0.0005,0.0001),(0.0005,0.0001),(0.0002,0.0001),(0.0001,0.00005),(0.00005,0.000025)
       ]

# The populations used NOTE RS may require a higher population than GC and HGC, up to 7x
populations = [#16,32,64,128,256 bits
               32,32,32,32,32,      #nOV
               32,64,64,64,64,      #OV  
               64,64,64,128,256,    #nDOV
               64,128,196,256,384]   #nPOV
problems = itertools.product(["nov","ov","ndov","npov"],["gc","hgc","rs"],[16,32,64,128,256])
reg_dict = {"{}_{}_{}".format(c,e,problem_size) : r for (c,e,problem_size),r in zip(list(problems),regs)}
problems = itertools.product(["nov","ov","ndov","npov"],[16,32,64,128,256])
pop_dict = {"{}_{}".format(c,problem_size) : r for (c,problem_size),r in zip(list(problems),populations)}

compression_ratio = 0.8
# EC problems do not support gpu, must keep as cpu
device = torch.device("cpu")

# Change this to get different combinations of compressions, environments, and sizes
# This is expecting one compression, one environment, and multiple sizes for graphing,
# although this could be slightly adapted to run multiple compression and environments
# Note that the maximum problem size that is supported in this script is up to 256 as 
# l1 and l2 values for larger sizes have not been calculated.
sizes = [16,32,64,128,256]
problems = itertools.product(["nov"],["gc"],sizes)

evals = []
for c, e, problem_size in problems:
    change_tolerance = problem_size * 3
    # Generate problem and params
    print(c,e,problem_size)
    problem_string = "{}_{}_{}".format(c,e,problem_size)
    problem = ECProblem(problem_size, c, e)
    print("Max possible fitness: {}".format(problem.max_fitness))
    l1_coef, l2_coef = reg_dict[problem_string]
    pop_size = pop_dict["{}_{}".format(c,problem_size)] 

    # Create model and population
    model = DOAE(problem_size, 0.2, device)
    # model = DOVAE(problem_size, round(problem_size*0.8), device)
    hidden_size = problem_size
    handler = OptimAEHandler(model, problem, device)
    population, fitnesses = handler.generate_population(pop_size)
    population, fitnesses, _, done = handler.hillclimb(population, fitnesses, change_tolerance)
    handler.print_statistics(fitnesses)

    max_depth = 6

    total_evals = 0
    depth = 0
    while True:
        # Do transition
        if depth < max_depth:
            hidden_size = round(compression_ratio * hidden_size)
            model.transition(hidden_size)
            depth += 1
            optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=l2_coef)
        handler.learn_from_population(population, optimizer, l1_coef=l1_coef, epochs=400)
        population, fitnesses, evaluations, done = handler.optimise_solutions(
            population, fitnesses, change_tolerance, encode=True
        )
        total_evals += evaluations
        population, fitnesses, evaluations, done = handler.hillclimb(
            population, fitnesses, change_tolerance
        )
        handler.print_statistics(fitnesses)
        total_evals += evaluations
        print("Evaluations: {}".format(total_evals))
        if done:
            break
    evals.append(total_evals)

print(evals)

plt.scatter(sizes, evals, marker="o")
plt.yscale("log")
plt.xscale("log")
plt.xticks(ticks=sizes, labels=sizes)
plt.minorticks_off()
plt.xlabel("Problem Size")
plt.ylabel("Evaluations")
plt.show()
