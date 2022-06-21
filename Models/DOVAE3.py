import random
import torch
from torch import nn
from torch.nn.modules.activation import Tanh
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

from COProblems.OptimizationProblem import HTOP, MCParity, OptimizationProblem, TestProblem

# Hyper-parameters
steps = 128
batch_size = 16
window_size = 32
lr = 0.001
dropout_prob = 0.0

class EncoderDistBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mean_layer = nn.Linear(input_size, output_size)
        self.logvar_layer = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        mu = self.mean_layer(x)
        logvar = self.logvar_layer(x)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        z = mu + eps*std

        return z, mu, logvar

class EncoderResidualBlock(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = []
        self.gates = []
        for i, size in enumerate(sizes[1:]):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
            self.gates.append(torch.zeros(sizes[i+1]).uniform_(-0.01, 0.01))
    
    def forward(self, x, zs):
        for z, layer, gate in zip(zs, self.layers, self.gates):
            x = torch.add(layer(x), z.matmul(torch.diag(gate)))
            x = F.tanh(x)
        return x


class DeepOptimizer(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.encoders = []
        self.decoder = nn.Sequential(
        )
        self.input_size = input_size
        self.sizes = [input_size]
    
    def forward(self, x):
        mus, logvars, zs = [], [], []
        for encoder in self.encoders:
            a = encoder[0](x, zs)
            z, mu, logvar = encoder[1](a)
            mus.append(mu)
            logvars.append(logvar)
            zs.append(z)
        if zs:
            x = zs[-1]
        x = self.decode(x)
        return x, mus, logvars
    
    # Passes data through the decoder at a particular layer
    def decode(self, x):
        return self.decoder(x)

    def add_layer(self, hidden_size):    
        self.encoders.append(
            nn.Sequential(
                EncoderResidualBlock(self.sizes),
                EncoderDistBlock(self.sizes[-1], hidden_size)
            )
        )
        decoder_layer = nn.Linear(hidden_size, self.sizes[-1])
        self.decoder = nn.Sequential(*([decoder_layer,nn.Tanh()] + list(self.decoder)))
        self.sizes.append(hidden_size)
            

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mus, logvars):
    MSE = F.mse_loss(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = 0
    for mu, logvar in zip(mus, logvars):
        KLD += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE, KLD

class PopulationDataset(Dataset):
    def __init__(self, X):
        self.X = X
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index]

def optimize_solution(s, model : DeepOptimizer, problem : OptimizationProblem):
    model.eval()
    solution = s[0]
    fitness = s[1]

    # H = Encode(S,L)
    hidden_repr, logvar = None, None

    if model.encoders:
        _, hidden_reprs, logvars = model(torch.tensor(solution, dtype=torch.float32))
        hidden_repr, logvar = hidden_reprs[-1], logvars[-1]
        # print("*****************")
        # print(hidden_repr)
        # print(logvar.exp())
    else:
        hidden_repr, _, _ = model(torch.tensor(solution, dtype=torch.float32))
    
    for _ in range(10000):
        
        # ii = [random.randint(0,len(hidden_repr)-1) for _ in range(1)]
        # # H' = H + ∆H
        # new_hidden_repr = hidden_repr.clone().detach()
        # for i in ii:
        #     new_hidden_repr[i] = random.choice([-1.0, 1.0])
        new_hidden_repr = hidden_repr.clone().detach()
        i = random.randint(0,len(hidden_repr)-1)
        if logvar is not None:
            std = torch.exp(0.5*logvar)
            std = 1 - std
            new_hidden_repr = torch.normal(new_hidden_repr, std)

            # multiplier = random.choice([1.0,-1.0])
            # new_hidden_repr[i] = new_hidden_repr[i] + (1 * multiplier * std[i])

            # Variables with a smaller std will be varied more
            # sft = 3 * F.softmax(-std, dim=0)
            # rand = torch.rand(sft.shape)
            # step = torch.randint(-1,2,std.shape, dtype=torch.float32)
            # steps = torch.where(sft > rand, step, torch.zeros(step.shape, dtype=torch.float32))
            # new_hidden_repr += steps

            # 1-std seems to perform best (?) rather than max-std
            # std = F.dropout(torch.max(std)-std, 0.8) * torch.randint(-1,2,std.shape, dtype=torch.float32)
            # new_hidden_repr += std
            
        else:
            new_hidden_repr[i] = random.choice([-1.0,1.0])
        
        # Sr = u(Decode(H))
        old_reconstruction = torch.sign(model.decode(hidden_repr))
        # S'r = u(Decode(H'))
        new_reconstruction = torch.sign(model.decode(new_hidden_repr))
        # ∆S = S'r − Sr
        delta_s = new_reconstruction - old_reconstruction

        #S' = S + ∆S
        new_solution = torch.tensor(solution, dtype=torch.float32) + delta_s
        new_solution = to_int_list(new_solution)
        # new_solution = model.decode(new_hidden_repr)
        # new_solution = to_int_list(new_solution)

        new_fitness = problem.fitness(new_solution)

        if new_fitness >= fitness:
            fitness = new_fitness
            solution = new_solution
            hidden_repr = new_hidden_repr
    
    return (solution, fitness)

def learn_from_population(model : DeepOptimizer, population, optimizer):
    training_set = list(map(lambda x : x[0], population))
    epochs = 100
    for epoch in range(epochs):
        dataset = DataLoader(PopulationDataset(training_set), batch_size=batch_size, shuffle=True)
        for i,x in enumerate(dataset):
            loss = learn_from_sample(model, x, optimizer, 0.01)
            print("Epoch {}/{} - {}/{} - Loss = {}".format(epoch+1,epochs,i,len(training_set),loss))

def learn_from_sample(model : DeepOptimizer, samples, optimizer, weight):
    xs = torch.stack(list(map(lambda x : torch.tensor(x,dtype=torch.float32), samples))).transpose(0,1)
    model.train()
    output, mus, logvars = model(F.dropout(xs, dropout_prob))
    MSE, KLD = loss_function(output, xs, mus, logvars)
    loss = MSE + weight * KLD
    # L1 loss
    # l1_lambda = 0.01
    # l1_norm = sum(p.abs().sum() for p in model.parameters())
    # loss = loss + l1_lambda * l1_norm

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return MSE.item()#loss.item()

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
    problem = HTOP(64)
    population = [problem.random_solution() for _ in range(steps)]
    population = list(map(lambda x : (x, problem.fitness(x)), population))
    model = DeepOptimizer(64)
    layer_sizes = [38,23,14]

    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)
    optimizer = None

    for layer in layer_sizes:
        print("Optimising solutions")
        cumulative_fitness = 0
        max_fitness = 0
        for i,sf in enumerate(population):
            
            with torch.no_grad():
                s,f = optimize_solution(sf, model, problem)

            cumulative_fitness += f
            if f > max_fitness:
                max_fitness = f
            if (i+1) % window_size == 0:
                print("Progress: {}/{} - Av. fitness: {} - Max fitness: {}".format(
                    i+1,steps,
                    cumulative_fitness/window_size,
                    max_fitness
                ))
                cumulative_fitness = 0
                max_fitness = 0
                
            population[i] = (s,f)
        
        #print(population)

        model.add_layer(layer)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
        learn_from_population(model, population, optimizer)


    for i,sf in enumerate(population):
            with torch.no_grad():
                s,f = optimize_solution(sf, model, problem)

            cumulative_fitness += f
            if f > max_fitness:
                max_fitness = f
            if (i+1) % window_size == 0:
                print("Progress: {}/{} - Av. fitness: {} - Max fitness: {}".format(
                    i+1,steps,
                    cumulative_fitness/window_size,
                    max_fitness
                ))
                cumulative_fitness = 0
                max_fitness = 0
                
            population[i] = (s,f)