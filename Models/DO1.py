import random
import torch
from torch import nn
from torch.nn.modules.activation import Tanh
import numpy as np
import matplotlib.pyplot as plt

from COProblems.OptimizationProblem import HTOP, MCParity, OptimizationProblem, TestProblem

# Hyper-parameters
change_tolerance = 5000
steps = 1500
batch_size = 1
window_size = 50
early_break_loss = 0.5
lr = 0.001
dropout_prob = 0.3

class DeepOptimizer(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_prob)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Tanh()
        )

        self.num_layers = 1
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    # Passes data through the decoder at a particular layer
    def run_decoder(self, layer, x):
        return self.decoder[-(layer-1)*3 - 2:](x)
    
    # Gets the size of the latent space at a particular layer
    def get_hidden_size(self, layer):
        return self.decoder[-(layer-1)*3 - 2].in_features

    def get_num_layers(self):
        return self.num_layers

    def add_layer(self, hidden_size):    

        param1 = nn.Linear(self.encoder[-3].out_features, hidden_size)
        param2 = nn.Linear(hidden_size, self.decoder[0].in_features)

        self.encoder = nn.Sequential(*(list(self.encoder) + [param1,nn.Tanh(),nn.Dropout(dropout_prob)]))
        self.decoder = nn.Sequential(*([param2,nn.Tanh(),nn.Dropout(dropout_prob)] + list(self.decoder)))

        self.num_layers += 1
    
    
    
# Convert tensor of continuous float values to python list of -1s and 1s
def to_int_list(x):
    x = torch.sign(x)
    x = x.tolist()
    x = [int(i) for i in x]
    return x

def learn_from_solution(model, criterion, optimizer, x):
    model.train()
    x = torch.tensor(x, dtype=torch.float32)
    # ===================forward=====================
    output = model(x)
    loss = criterion(output, x)
    # ===================backward====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Optimize solution via model informed variation
def optimize_solution(model : DeepOptimizer, hidden_repr, problem : OptimizationProblem,
                      current_solution, prev_layer : bool):
    # No gradient needed as the model isn't learning from these samples
    with torch.no_grad():
        # Disable dropout layers
        model.eval()
        current_solution = to_int_list(current_solution)
        current_fitness = problem.fitness(current_solution)  
        # Continue applying variation until a certain number of steps of no fitness improvements occuring
        no_change_counter = 0
        #while True:
        for _ in range(20):
            # Loop through every index of latent representation in random order
            indices = [i for i in range(len(hidden_repr))]
            random.shuffle(indices)
            for i in indices:
                # Flip bit and determine whether this has improved fitness
                hidden_repr[i] *= -1
                new_solution = model.run_decoder(model.get_num_layers() - prev_layer, hidden_repr)
                new_solution = to_int_list(new_solution)
                new_fitness = problem.fitness(new_solution)
                if new_fitness >= current_fitness:
                    if new_fitness == current_fitness:
                        no_change_counter += 1
                    else:
                        no_change_counter = 0
                    current_fitness = new_fitness
                    current_solution = new_solution
                # If new fitness is not better, reset the flipped bit to what it was before
                else:
                    hidden_repr[i] *= -1
                    no_change_counter += 1
            if no_change_counter > change_tolerance:
                break
    return current_solution, hidden_repr

def optimize_solution2(model : DeepOptimizer, hidden_repr, problem : OptimizationProblem,
                      current_solution, prev_layer : bool):
    # No gradient needed as the model isn't learning from these samples
    with torch.no_grad():
        # Disable dropout layers
        model.eval()
        current_solution = to_int_list(current_solution)
        current_fitness = problem.fitness(current_solution)  
        for _ in range(2000):
            indices = [random.randint(0,len(hidden_repr)-1) for _ in range(5)]
            #i = random.randint(0,len(hidden_repr)-1)
            # Flip bit and determine whether this has improved fitness
            for i in indices:
                hidden_repr[i] *= -1
            new_solution = model.run_decoder(model.get_num_layers() - prev_layer, hidden_repr)
            new_solution = to_int_list(new_solution)
            new_fitness = problem.fitness(new_solution)
            if new_fitness >= current_fitness:
                current_fitness = new_fitness
                current_solution = new_solution
            # If new fitness is not better, reset the flipped bit to what it was before
            else:
                for i in indices:
                    hidden_repr[i] *= -1

    return current_solution, hidden_repr

    

# Standard hillclimbing algorithm
def hillclimb(problem : OptimizationProblem):
    solution = problem.random_solution()
    fitness = problem.fitness(solution)
    old_fitness = fitness
    for _ in range(1000):
        index = random.randint(0,len(solution) - 1)
        solution[index] *= -1
        new_fitness = problem.fitness(solution)
        if fitness <= new_fitness:
            fitness = new_fitness
        # If new fitness is not better, reset the flipped bit to what it was before
        else:
            solution[index] *= -1
    return solution

def learn(problem : OptimizationProblem, model : DeepOptimizer, criterion, optimizer,
          use_hillclimb : bool, early_break : bool, prev_layer : bool):
    print("Generating solutions")
    cumulative_fitness = 0
    max_fitness = 0
    total_loss = 0
    samples = []
    best_epoch_solution = []
    best_epoch_fitness = 0
    for i in range(steps):
        optimized_solution = None
        if use_hillclimb:
            optimized_solution = hillclimb(problem)
        else:
            # Generate new solutions and optimize according to model informed variation
            hidden_repr = torch.sign(torch.randn(model.get_hidden_size(model.get_num_layers()-prev_layer)))
            solution = model.run_decoder(model.get_num_layers()-prev_layer, hidden_repr)
            optimized_solution, _ = optimize_solution2(model, hidden_repr, problem, solution, prev_layer)
        # Append optimized solution to batch
        samples.append(torch.tensor(optimized_solution,dtype=torch.float32))
        fitness = problem.fitness(optimized_solution)
        cumulative_fitness += fitness
        max_fitness = max(fitness, max_fitness)

        if fitness > best_epoch_fitness:
            best_epoch_fitness = fitness
            best_epoch_solution = optimized_solution
        if (i+1) % batch_size == 0:
            total_loss += learn_from_solution(model, criterion, optimizer, torch.stack(samples))
            samples = []
        if (i+1) % window_size == 0:
            print("Progress: {}/{} - Av. fitness: {} - Max fitness: {} - Av. loss: {}".format(
                i+1,steps,
                cumulative_fitness/window_size,
                max_fitness,
                total_loss/(window_size/batch_size))
            )
            if (total_loss*batch_size)/window_size < early_break_loss and early_break:
                break
            cumulative_fitness = 0
            max_fitness = 0
            total_loss = 0
    print("Best solution: {}".format(best_epoch_solution))

def get_best_latent_repr(model : DeepOptimizer, problem : OptimizationProblem):
    solutions = []
    for _ in range(100):
        hidden_repr = torch.sign(torch.randn(model.get_hidden_size(model.get_num_layers())))
        solution = model.run_decoder(model.get_num_layers(), hidden_repr)
        optimized_solution, hidden_repr = optimize_solution(model, hidden_repr, problem, solution, False)
        solutions.append((problem.fitness(optimized_solution), hidden_repr))
    solutions.sort(key=lambda x : x[0], reverse=True)
    return solutions[:2]

def graph_latent(model : DeepOptimizer, problem : OptimizationProblem):
    solutions = get_best_latent_repr(model, problem)
    f1, latent1 = solutions[0]
    f2, latent2 = solutions[1]
    points = np.linspace(latent1, latent2, 50)
    model.eval()
    fitnesses = list(map(lambda p : problem.fitness(to_int_list(model.decoder(torch.tensor(p)))), points))
    print("Distance between points: {}".format(np.linalg.norm(latent1 - latent2)))
    plt.plot([i for i in range(len(points))], fitnesses)
    plt.show()


model = DeepOptimizer(64, 58)
layer_lengths = [53,48]
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)
#optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

# Change this to appropriate problem that inherits from OptimizationProblem
problem = HTOP(4)

learn(problem, model, criterion, optimizer, True, True, True)
#graph_latent(model, problem)
for i,hidden_size in enumerate(layer_lengths):
    model.add_layer(hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    print("New layer added successfully")
    learn(problem, model, criterion, optimizer, False, True, True)
    #graph_latent(model, problem)

learn(problem, model, criterion, optimizer, False, False, False)
graph_latent(model, problem)

