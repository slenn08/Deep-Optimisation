import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import sys
sys.path.append(".")
from COProblems.OptimisationProblem import ECProblem
from OptimAE import OptimAEHandler

problem_size = 64
compression = "npov"
environment = "gc"
pop_size = 128
problem = ECProblem(problem_size,compression,environment)

handler = OptimAEHandler(None, problem)
population, fitnesses = handler.generate_population(pop_size)
population, fitnesses, _, _ = handler.hillclimb(population, fitnesses, 64)

lr = 0.001
batch_size = 16

points_dim = 8
points_number = problem_size

encoder = torch.nn.Sequential(
    torch.nn.Linear(problem_size, points_dim),
    torch.nn.Tanh(), 
)
decoder = torch.nn.Sequential(
    torch.nn.Linear(points_dim, points_number),
    torch.nn.Tanh(),
    torch.nn.Linear(points_number, problem_size),
    torch.nn.Tanh()
)

optimizer = torch.optim.Adam([{"params": encoder.parameters()}, {"params": decoder.parameters()}],
                             lr=lr)

# TODO try dot product between pos embedding and solution embedding instead of cdist
epochs = 500
for epoch in range(epochs):
    dataset = DataLoader(TensorDataset(population), batch_size=batch_size, shuffle=True)
    for i,x in enumerate(dataset):
        sol_emb = encoder(F.dropout(x[0], 0.2))
        recon = decoder(sol_emb)
        mse = F.mse_loss(recon, x[0])
        # distance_loss = 0.001 * torch.mean(distances)
        # l1 = 0.00001 * sum(p.abs().sum() for p in model.parameters())
        loss = mse #+ distance_loss #+ l1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch {}/{} - {}/{} - Loss = {}".format(epoch+1,epochs,i,len(population),mse))

handler.print_statistics(fitnesses)
i = fitnesses.argmax(dim=0)
solution = population[i]
print(fitnesses[i])
latents = encoder(solution)
sol_embs = latents.reshape((1,points_dim))
old_recon = torch.sign(decoder(sol_embs)).reshape((problem_size))

print(latents)

fig, axes = plt.subplots(ncols=4, figsize=(16,4))
for i ,(ax, latent) in enumerate(zip(axes, latents)): 
    l = latents.clone()
    new_fitnesses = []
    values = np.linspace(-1,1,1000)
    for x in values:
        l[i] = x
        new_sol_embs = l.reshape((1,points_dim))
        new_recon = decoder(new_sol_embs)
        new_recon = torch.sign(new_recon).reshape((problem_size))
        delta_s = new_recon - old_recon
        new_s = torch.sign(solution + delta_s)

        fitness = problem.fitness(new_s.detach().numpy().reshape((problem_size)))
        new_fitnesses.append(fitness)
    ax.plot(values,new_fitnesses)
    
plt.show()
