import random
import torch
from torch import nn
from torch.nn.modules.activation import Tanh
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import weight_norm
from torch import linalg

import sys
sys.path.append(".")

from COProblems.OptimizationProblem import HTOP, MKP, ECProblem, MCParity, OptimizationProblem, TestProblem
from Experiments.data import data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeepOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.mean_layer = None
        self.std_layer =None
        self.decoder = None
        self.input_size = input_size
        self.hidden_size = hidden_size
    
    def encode(self, x):
        return self.mean_layer(x), self.std_layer(x)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z
    
    def decode(self, x):
        return self.decoder(x)

    def reset_weights(self):    
        self.mean_layer = weight_norm(nn.Linear(self.input_size, self.hidden_size), name='weight')
        self.std_layer = weight_norm(nn.Linear(self.input_size, self.hidden_size), name='weight')      
        decoder_layer = weight_norm(nn.Linear(self.hidden_size, self.input_size), name='weight')
        self.decoder = nn.Sequential(decoder_layer,nn.Tanh())

    def SR(self, sr_weight):
        sr = 0
        for layer in [self.mean_layer, self.std_layer, self.decoder[0]]:
            weight = layer.weight_v.transpose(0,1).matmul(layer.weight_g).transpose(0,1)
            _,sv,_ = linalg.svd(weight)
            sr += torch.max(sv)

        return sr * sr_weight

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    recon_MSE = F.mse_loss(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # (This has been changed from sum to mean to scale it properly)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_MSE, KLD

class PopulationDataset(Dataset):
    def __init__(self, X):
        # Extract solutions from population
        self.X = list(map(lambda x : x[0], X))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return {"solution" : self.X[index]}

def random_indices(l):
    ii = [i for i in range(len(l))]
    random.shuffle(ii)
    return ii

def hillclimb_optimize(s, problem : OptimizationProblem):
    solution = s[0]
    ii = random_indices(solution)
    fitness = s[1]
    while True:
        i = ii.pop()
        solution[i] *= -1
        new_fitness = problem.fitness(to_int_list(solution))
        if new_fitness > fitness:
            fitness = new_fitness
            ii = random_indices(solution)
        if new_fitness < fitness:
            # Reset solution if negative fitness change
            solution[i] *= -1
        if len(ii) == 0:
            return (solution,fitness)

def encode_step(s, model : DeepOptimizer, v):
    s = s.clone()
    #H = Encode(Solution,L);
    h, logvar = model.encode(s)
    std = torch.exp(0.5*logvar)
    #Hs = Encode(Solution+∆S,L);
    d_s_index = torch.randint(0,s.shape[1], (1,))
    s[:,d_s_index] *= -1
    hs, _ = model.encode(s)
    # ∆H = Hs − H ;
    d_h = hs - h
    # d_h = torch.sign(d_h) * std * 2
    # a = mean(|∆H|);
    a = torch.mean(torch.abs(d_h), dim=1)
    # z = max(|∆Hi|) ;
    z,_ = torch.max(torch.abs(d_h), dim=1)
    # T = a + (z − a) × U(0, 1) ;
    t = a + (z - a) * torch.rand(z.shape)

    # # topk, indices = torch.topk(torch.abs(d_h), v, dim=1)
    # # d_h = torch.zeros(d_h.shape).scatter(1, indices, d_h)
    # # d_h = torch.sign(d_h)*std*5
    d_h = torch.where(torch.abs(d_h) > t.reshape((d_h.shape[0],1)), torch.sign(d_h)*std*5, torch.zeros_like(d_h))
    return d_h

def sample(pop_size, latent_size, model : DeepOptimizer, problem : OptimizationProblem):
    samples = torch.randn((pop_size*2, latent_size))
    population_raw = model.decode(samples)
    population = []
    for p in population_raw:
        f = problem.fitness(to_int_list(p))
        population.append((p,f))
    population = sorted(population, key=lambda x:x[1])
    population = population[len(population)//2:]
    return population


def optimize_solutions_bulk(population, model : DeepOptimizer, problem : OptimizationProblem, no_change_prob, change_tolerance, latent_size):
    solutions = torch.stack(list(map(lambda x : x[0], population))).to(device)
    fitnesses = torch.tensor(list(map(lambda x : x[1], population))).to(device)
    evaluations = 0
    last_improve = torch.zeros_like(fitnesses)
    v = round(latent_size * 0.05)

    while True:
        hidden_repr, logvar = model.encode(solutions)
        new_hidden_repr = hidden_repr.clone().detach()
        std = torch.exp(0.5*logvar)

        # step = torch.normal(torch.zeros_like(std, device=device), (std_multiplier)*std)
        # step = torch.normal(torch.zeros_like(std, device=device), 5*std)
        # step = torch.where(torch.abs(step) > 5*std, step, 5*std)
        # step = F.dropout((1-no_change_prob) * step, no_change_prob)
        # new_hidden_repr = new_hidden_repr + step
        
        # indices = torch.randint(0,hidden_repr.shape[1], (hidden_repr.shape[0],))
        # step = std[torch.arange(hidden_repr.shape[0]), indices]*10
        # direction = torch.randint(0,2,(hidden_repr.shape[0],)) * 2 - 1
        # new_hidden_repr[torch.arange(hidden_repr.shape[0]), indices] += (step * direction) 
        # new_solutions = solutions.clone()
        # unfinished_indices = torch.arange(0,solutions.shape[0],1)   
        # while (len(unfinished_indices) > 0):
        #     indices = torch.randint(0,hidden_repr.shape[1], (v,))    
        #     step = std[:, indices][unfinished_indices] * 5
        #     direction = torch.randint(0,2,step.shape) * 2 - 1
        #     step *= direction
        #     #new_hidden_repr[:, indices] += step
        #     masked = new_hidden_repr[unfinished_indices]
        #     masked[:,indices] += step
        #     new_hidden_repr[unfinished_indices] = masked
        #     unfinished_indices = torch.where((new_solutions == solutions).all(dim=1))[0]
           
            #print(new_hidden_repr - old_new_hidden_repr)
            # new_hidden_repr += encode_step(solutions, model, v)

        indices = torch.randint(0,hidden_repr.shape[1], (v,))    
        step = std[:, indices] * 5
        direction = torch.randint(0,2,step.shape) * 2 - 1
        step *= direction
        new_hidden_repr[:, indices] += step
        # Sr = u(Decode(H))
        old_reconstruction = torch.sign(model.decode(hidden_repr))
        # S'r = u(Decode(H'))
        new_reconstruction = torch.sign(model.decode(new_hidden_repr))
        # ∆S = S'r − Sr
        delta_s = new_reconstruction - old_reconstruction
        #S' = S + ∆S
        new_solutions = torch.sign(solutions + delta_s)

        for i, (solution, new_solution, fitness) in enumerate(zip(solutions, new_solutions, fitnesses)):
            if torch.equal(solution, new_solution) or last_improve[i] > change_tolerance:
                last_improve[i] += 1
                continue
            #new_solution = problem.repair(new_solution)
            new_fitness = problem.fitness(to_int_list(new_solution))
            evaluations += 1

            if new_fitness >= fitness:
                if new_fitness > fitness:
                    last_improve[i] = 0 
                else:
                    last_improve[i] += 1
                solutions[i] = new_solution
                fitnesses[i] = new_fitness       
                if new_fitness == problem.max_fitness:
                    return (solutions, fitnesses.tolist(), evaluations)  
            else:
                last_improve[i] += 1

        if torch.all(last_improve > change_tolerance):
            return (solutions, fitnesses.tolist(), evaluations)            

        
def optimize_solution(s, model : DeepOptimizer, problem : OptimizationProblem, std_multiplier,
                      n, no_change_prob, change_tolerance):
    model.eval()
    solution = s[0]
    fitness = s[1]
    evaluations = 0
    last_improve = 0

    hidden_repr, logvar = model.encode(solution)
    
    while True:
        new_hidden_repr = hidden_repr.clone().detach()

        std = torch.exp(0.5*logvar)

        # step = torch.normal(torch.zeros_like(std), 5*std)

        step = torch.normal(torch.zeros_like(std, device=device), (std_multiplier)*std)
        step = F.dropout((1-no_change_prob) * step, no_change_prob)

        new_hidden_repr = new_hidden_repr + step

        # i = random.randint(0,len(hidden_repr)-1)
        # new_hidden_repr[i] = torch.normal(new_hidden_repr[i], std[i] * std_multiplier[i])
        # new_hidden_repr[i] += std[i] * std_multiplier

        # Sr = u(Decode(H))
        old_reconstruction = torch.sign(model.decode(hidden_repr))
        # S'r = u(Decode(H'))
        new_reconstruction = torch.sign(model.decode(new_hidden_repr))
        # ∆S = S'r − Sr
        delta_s = new_reconstruction - old_reconstruction
        #S' = S + ∆S
        new_solution = torch.sign(solution + delta_s)
        
        if not torch.equal(new_solution, solution):
            new_fitness = problem.fitness(to_int_list(new_solution))
            evaluations += 1

            if new_fitness >= fitness:
                if new_fitness > fitness:
                    last_improve = 0
                    step[step > 0] = step[step > 0].abs() / std[step > 0]
                    s_n = n[step>0]
                    std_multiplier[step>0] = std_multiplier[step>0] * ((s_n-1)/s_n) + step[step > 0]/s_n
                    n[step > 0] += 1

                    #step[i] = step[i].abs() / std[i]
                    #std_multiplier[i] = std_multiplier[i] * ((n[i]-1)/n[i]) + step[step > 0]/n[i]
                    # n[i] += 1

                    # std_multiplier = std_multiplier * ((n-1)/n) + std_sum/n 
                        
                else:
                    last_improve += 1
                fitness = new_fitness
                solution = new_solution
                hidden_repr, logvar = model.encode(solution)
            else:
                last_improve += 1
        else:
            last_improve += 1
            # hidden_repr = new_hidden_repr

        if last_improve > change_tolerance:
            return (solution, fitness, evaluations, std_multiplier, n)

def learn_from_population(model : DeepOptimizer, population, optimizer, batch_size, latent_size, kl_weight):
    epochs = 400
    for epoch in range(epochs):
        dataset = DataLoader(PopulationDataset(population), batch_size=batch_size, shuffle=True)
        for i,x in enumerate(dataset):
            loss = learn_from_sample(model, x["solution"], optimizer, kl_weight)
            # print("Epoch {}/{} - {}/{} - Loss = {}".format(epoch+1,epochs,i,len(population),loss))
    # dataset = DataLoader(PopulationDataset(population), batch_size=5, shuffle=True)
    # x = next(iter(dataset))
    # show_mu_sd(model, x["solution"])

def show_mu_sd(model, xs):
    for x in xs:
        mu, logvar = model.encode(x)
        std = torch.exp(0.5*logvar)
        print("***************************************************")
        print(mu)
        print(std)


def learn_from_sample(model : DeepOptimizer, samples, optimizer, kl_weight):
    #xs = torch.stack(samples)
    xs = samples
    model.train()
    output, mu, logvar, z = model(xs)
    MSE, KLD = loss_function(output, xs, mu, logvar)
    loss = MSE + kl_weight * KLD #+ model.SR(0.001)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return MSE.item()

# Convert tensor of continuous float values to python list of -1s and 1s
def to_int_list(x):
    try:
        x = torch.sign(x)
        x = x.tolist()
        x = [int(i) for i in x]  
    except TypeError:
        pass
    return x

def optimize_population_hillclimb(population, problem : OptimizationProblem, window_size, pop_size):
    cumulative_fitness = 0
    max_fitness = 0
    for i,sf in enumerate(population):
        s,f = hillclimb_optimize(sf, problem)
        if f == problem.max_fitness:
            print("Optimal found in hillclimb step")
            print(s)
            quit()

        cumulative_fitness += f
        if f > max_fitness:
            max_fitness = f
        if (i+1) % window_size == 0:
            print("Progress: {}/{} - Av. fitness: {} - Max fitness: {}".format(
                i+1,pop_size,
                cumulative_fitness/window_size,
                max_fitness,
            ))
            cumulative_fitness = 0
            max_fitness = 0
        population[i] = (s,f)
    return population

def optimize_population_model(population, model, problem, total_eval, std_mul_prior, latent_size,
                              no_change_prob, change_tolerance, window_size, pop_size):
    n = torch.full((latent_size,), 1, device=device)
    std_multiplier = torch.full((latent_size,), std_mul_prior, device=device)
    cumulative_fitness = 0
    max_fitness = 0

    with torch.no_grad():
        s,f,evaluations = optimize_solutions_bulk(population, model, problem, no_change_prob, change_tolerance, latent_size)
        
        total_eval += evaluations
    # population = list(zip(s,f))
    # population = optimize_population_hillclimb(population, problem, window_size, pop_size)
    # s = list(map(lambda sf: sf[0], population))
    # f = list(map(lambda sf: sf[1], population))
    if problem.max_fitness in f:
        print("Optimal found in {} steps".format(total_eval))
        return [], total_eval, True

    cumulative_fitness += sum(f)
    for fitness in f:
        if fitness > max_fitness:
            max_fitness = fitness
    print("Av. fitness: {} - Max fitness: {} - Evaluations: {}".format(
        cumulative_fitness/pop_size,
        max_fitness,
        total_eval,
    ))
    return list(zip(s,f)), total_eval, False

def get_best_in_pop(population):
    sorted_pop = sorted(population, key=lambda x : x[1], reverse=True)
    return sorted_pop[:len(sorted_pop)//4]
    

if __name__ == "__main__":
    # Hyper-parameters
    
    batch_size = 16
    window_size = 32
    lr = 0.001
    kl_weight = 0.1
    compression = 0.8
    no_change_prob = 0.95
    pop_size = 1000
    problem_size = 100
    change_tolerance = 100
    problem = MKP("COProblems\\mkp\\problems10d.txt", 5)
    problem.max_fitness = 22777
    # change_tolerance = 128
    # pop_size = 512
    # problem_size = 256
    # c = "ov"
    # e = "rs"
    # problem = ECProblem(problem_size, c, e)
    latent_size = int(compression*problem_size)
    attempts = 1
    mean_evals = 0

    for _ in range(attempts):
        model = DeepOptimizer(problem_size, latent_size)
        model.to(device)
        population = [problem.random_solution() for _ in range(pop_size)]
        population = list(map(lambda x : (torch.tensor(x, dtype=torch.float32, device=device), problem.fitness(x)), population))
        population = optimize_population_hillclimb(population, problem, window_size, pop_size)

        # Convert to pytorch tensor, apply fitness to data
        # population = data["{}_{}_{}".format(c,e,problem_size)][:pop_size]
        # population = list(map(lambda x : (torch.tensor(x[0], dtype=torch.float32, device=device), x[1]), population))
        # print(population)    
        total_evaluations = 0
        
        population_residue = [(s.clone(), f) for s,f in population]
        model.reset_weights()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        learn_from_population(model, population, optimizer, batch_size, latent_size, kl_weight)
        
        while True:
            print("Optimising solutions")
            population, total_evaluations, solved = optimize_population_model(population, model, problem,
                                                                    total_evaluations,3.0, latent_size,
                                                                    no_change_prob,change_tolerance,window_size,
                                                                    pop_size)
            if solved:
                mean_evals += total_evaluations
                break
            
            # population_residue += population[:pop_size//4]
            # population_residue = population_residue[-pop_size:]

            model.reset_weights()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # learn_from_population(model, population+population_residue, optimizer, batch_size, latent_size, kl_weight)
            learn_from_population(model, population, optimizer, batch_size, latent_size, kl_weight)

    mean_evals /= attempts
    print("Mean evals: {}".format(mean_evals))
    