import sys
sys.path.append(".")
from COProblems.OptimizationProblem import ECProblem, generate_linkage
from Models.DOVAE2 import hillclimb_optimize
pop_size = 256
data = {}
linkages = {}
with open("data2.txt", mode="w") as file:
    for c in ["nov", "ov", "ndov", "npov"]:
        for e in ["gc", "hgc", "rs"]:
            for problem_size in [16,32,64,128,256]:    
                print(c,e,problem_size)
                linkage = None
                if e == "hgc":
                    linkage = generate_linkage(problem_size)
                problem = ECProblem(problem_size, c, e, linkage)
                problem.max_fitness = float('inf')
                population = [problem.random_solution() for _ in range(pop_size)]
                population = list(map(lambda x: (x, problem.fitness(x)), population))
                for i,s in enumerate(population):
                    population[i] = hillclimb_optimize(s, problem)
                data["{}_{}_{}".format(c,e,problem_size)] = population
                linkages["{}_{}_{}".format(c,e,problem_size)] = linkage
    file.write("data = {}\n".format(data))
    file.write("linkages = {}\n".format(linkages))