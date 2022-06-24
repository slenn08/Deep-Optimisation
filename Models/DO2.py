import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from COProblems.OptimizationProblem import HTOP, ECProblem, MCParity, OptimizationProblem, TestProblem
from Experiments.data import data
# Hyper-parameters


batch_size = 16
window_size = 32
lr = 0.002
dropout_prob = 0.2
compression = 0.9
l1_reg = 0.0001
l2_reg = 0.00005
pop_size = 256
problem_size = 128
attempts = 1
change_tolerance = 128
c = "npov"
e = "hgc"
problem = ECProblem(problem_size, c, e)
encode = False

class DeepOptimizer(nn.Module):
    def __init__(self,input_size):
        super().__init__()

        self.encoder = nn.Sequential(nn.Dropout(dropout_prob))
        self.decoder = nn.Sequential()
        self.input_size = input_size   
        self.num_layers = 1
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
    
    def encode(self, x, layer):
        return self.encoder[:1+(2*layer)](x)

    def decode(self, x, layer):
        return self.decoder[(self.num_layers-layer-1)*2:](x)

    def add_layer(self, hidden_size):   
        prev_size = 0
        if not self.decoder:
            prev_size = self.input_size
        else: 
            prev_size = self.decoder[0].in_features

        weight = torch.tensor([[random.uniform(-0.01,0.01) for _ in range(prev_size)]
                                         for _ in range(hidden_size)], requires_grad=True)
        encoder_layer = nn.Linear(prev_size, hidden_size)
        encoder_layer.weight = nn.Parameter(weight)
        decoder_layer = nn.Linear(hidden_size, prev_size)
        decoder_layer.weight = nn.Parameter(weight.transpose(0,1))

        self.encoder = nn.Sequential(*(list(self.encoder) + [encoder_layer,nn.Tanh()]))
        self.decoder = nn.Sequential(*([decoder_layer,nn.Tanh()] + list(self.decoder)))

        self.num_layers += 1
    def reset_weights(self, hidden_size):
        self.encoder = nn.Sequential(nn.Dropout(dropout_prob))
        self.decoder = nn.Sequential()
        self.add_layer(hidden_size)
        if self.num_layers > 2:
            self.num_layers = 2
        

class PopulationDataset(Dataset):
    def __init__(self, X):
        self.X = X
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index]

def assign_step(s, model : DeepOptimizer, layer_num, encode=False):
    # H = Encode(S,L)
    hidden_repr = model.encode(torch.tensor(s, dtype=torch.float32), layer_num)
    new_hidden_repr = hidden_repr.clone().detach()

    if (not encode) or layer_num == 0:     
        i = random.randint(0,len(hidden_repr)-1)
        new_hidden_repr[i] = random.choice([-1.0,1.0])
    else:
        d_h = encode_step(s, model, layer_num)
        new_hidden_repr += d_h
    # Sr = u(Decode(H))
    old_reconstruction = torch.sign(model.decode(hidden_repr, layer_num))
    # S'r = u(Decode(H'))
    new_reconstruction = torch.sign(model.decode(new_hidden_repr, layer_num))
    # ∆S = S'r − Sr
    delta_s = new_reconstruction - old_reconstruction

    #S' = S + ∆S
    new_solution = torch.tensor(s, dtype=torch.float32) + delta_s
    new_solution = to_int_list(new_solution)
    return new_solution

def encode_step(s, model : DeepOptimizer, layer_num):
    # create copy of s
    s = s[:]
    #H = Encode(Solution,L);
    h = model.encode(torch.tensor(s, dtype=torch.float32), layer_num)
    #Hs = Encode(Solution+∆S,L);
    d_s_index = random.randint(0, len(s)-1)
    s[d_s_index] *= -1
    hs = model.encode(torch.tensor(s, dtype=torch.float32), layer_num)
    # ∆H = Hs − H ;
    d_h = hs - h
    # a = mean(|∆H|);
    a = torch.mean(torch.abs(d_h))
    # z = max(|∆Hi|) ;
    z = torch.max(torch.abs(d_h))
    # T = a + (z − a) × U(0, 1) ;
    t = a + (z - a) * torch.rand(1).item()
    
    d_h = torch.where(torch.abs(d_h) > t, torch.sign(d_h) - h, torch.zeros_like(d_h))
    return d_h
    
def optimize_solution(s, model : DeepOptimizer, problem : OptimizationProblem, encode=False):
    evaluations = 0
    model.eval()
    solution = s[0]
    fitness = s[1]
    #for _ in range(5000):
    for layer_num in range(model.num_layers-1, -1, -1):
        last_improve = 0
        while True:
            new_solution = assign_step(solution, model, layer_num, encode)

            if new_solution != solution:
                new_fitness = problem.fitness(new_solution)
                evaluations += 1

                if new_fitness >= fitness:
                    if new_fitness > fitness:
                        last_improve = 0
                    else:
                        last_improve += 1
                    fitness = new_fitness
                    solution = new_solution
                else:
                    last_improve += 1
            else:
                last_improve += 1
            if last_improve > change_tolerance:
                break
    return (solution, fitness, evaluations)           
    # hidden_repr = new_hidden_repr

def learn_from_population(model : DeepOptimizer, population, criterion, optimizer, l1_reg=0.0005):
    training_set = list(map(lambda x : x[0], population))
    epochs = 400
    for epoch in range(epochs):
        dataset = DataLoader(PopulationDataset(training_set), batch_size=batch_size, shuffle=True)
        for i,x in enumerate(dataset):
            loss = learn_from_sample(model, x, criterion, optimizer, l1_reg)
            #print("Epoch {}/{} - {}/{} - Loss = {}".format(epoch+1,epochs,i,len(training_set),loss))

def learn_from_sample(model : DeepOptimizer, samples, criterion, optimizer, l1_reg):
    xs = torch.stack(list(map(lambda x : torch.tensor(x,dtype=torch.float32), samples))).transpose(0,1)
    # xs = samples
    model.train()
    output, _ = model(xs)
    loss = criterion(output, xs)
    # L1 loss
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    loss = loss + l1_reg * l1_norm

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Convert tensor of continuous float values to python list of -1s and 1s
def to_int_list(x):
    try:
        x = torch.sign(x)
        x = x.tolist()
        x = [int(i) for i in x]  
    except TypeError:
        pass
    return x

if __name__ == "__main__":
    mean_evals = 0
    for _ in range(attempts):
        print("New simulation")
        #problem = HTOP(128)
        population = [problem.random_solution() for _ in range(pop_size)]
        population = list(map(lambda x : (x, problem.fitness(x)), population))
        # population = data["{}_{}_{}".format(c,e,problem_size)][:pop_size]
        model = DeepOptimizer(problem_size)
        layer_sizes = [int(problem_size*(compression**i)) for i in range(1,20)]

        criterion = nn.MSELoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)
        optimizer = None

        start_eval_count = False
        total_evaluations = 0
        finished = False

        for layer in layer_sizes:
        #while True:
            print("Optimising solutions")
            cumulative_fitness = 0
            max_fitness = 0
            for i,sf in enumerate(population):
                
                with torch.no_grad():
                    s,f,evaluations = optimize_solution(sf, model, problem, encode=encode)
                
                if start_eval_count:
                    total_evaluations += evaluations
                if f == problem.max_fitness:
                    print("Optimal found in {} steps".format(total_evaluations))
                    mean_evals += total_evaluations
                    finished = True
                    break

                cumulative_fitness += f
                if f > max_fitness:
                    max_fitness = f
                if (i+1) % window_size == 0:
                    print("Progress: {}/{} - Av. fitness: {} - Max fitness: {} - Evaluations: {}".format(
                        i+1,pop_size,
                        cumulative_fitness/window_size,
                        max_fitness,
                        total_evaluations
                    ))
                    cumulative_fitness = 0
                    max_fitness = 0
                    
                population[i] = (s,f)

            if finished:
                break
            model.add_layer(layer)
            #model.reset_weights(layer_sizes[0])
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
            learn_from_population(model, population, criterion, optimizer, l1_reg)
            start_eval_count = True
    print("Mean evaluations: {}".format(mean_evals/attempts))


    # for i,sf in enumerate(population):
    #         with torch.no_grad():
    #             s,f, evaluations = optimize_solution(sf, model, problem, encode=encode)
            
    #         total_evaluations += evaluations

    #         if f == problem.max_fitness:
    #             print("Optimal found in {} steps".format(total_evaluations))
    #             quit()

    #         cumulative_fitness += f
    #         if f > max_fitness:
    #             max_fitness = f
    #         if (i+1) % window_size == 0:
    #             print("Progress: {}/{} - Av. fitness: {} - Max fitness: {}".format(
    #                 i+1,pop_size,
    #                 cumulative_fitness/window_size,
    #                 max_fitness
    #             ))
    #             cumulative_fitness = 0
    #             max_fitness = 0
                
    #         population[i] = (s,f)