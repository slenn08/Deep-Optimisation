from copy import deepcopy
import random
from COProblems.OptimizationProblem import ECProblem

change_tolerance = 1024
size = 32
problem = ECProblem(size, "nov", "gc")
# solution = problem.random_solution()
# solution = [-1,1,-1,1,-1,1,-1,1]
# fitness = problem.fitness(solution)
# print(fitness)
solutions_passed = 0

while True:
    solution = problem.random_solution()
    fitness = problem.fitness(solution)
    changed_indices = []
    solution_timeline = [deepcopy(solution)]
    last_improve = 0
    evaluations = 0
    while True:
        i = random.randint(0, len(solution)-1)
        solution[i] *= -1
        new_fitness = problem.fitness(solution)
        evaluations += 1
        if new_fitness > fitness:
            fitness = new_fitness
            last_improve = 0
            changed_indices.append(i)
            solution_timeline.append(deepcopy(solution))
            if fitness == problem.max_fitness:
                print("Solution found in {} evaluations".format(evaluations))
                for i,s in zip(changed_indices, solution_timeline):
                    s[i] = "_"
                    print(" ".join(map(str,s)))
                #print(solution)
                quit()
        elif new_fitness == fitness:
            changed_indices.append(i)
            solution_timeline.append(deepcopy(solution))
            last_improve += 1
        else:
            last_improve += 1
            # Reset solution
            solution[i] *= -1
        
        if last_improve > change_tolerance:
            solutions_passed += 1
            print(solutions_passed)
            break