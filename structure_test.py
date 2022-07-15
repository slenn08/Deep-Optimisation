import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import sys
sys.path.append(".")
from COProblems.OptimisationProblem import ECProblem
from OptimAE import OptimAEHandler

problem_size = 128
compression = "nov"
environment = "gc"
pop_size = 128
problem = ECProblem(problem_size,compression,environment)

handler = OptimAEHandler(None, problem)
population, fitnesses = handler.generate_population(pop_size)
population, fitnesses, _, _ = handler.hillclimb(population, fitnesses, 64)

# population = data["nov_gc_256"][:pop_size]
# fitnesses = torch.tensor(list(map(lambda x : x[1], population)), dtype=torch.float32)
# population = torch.tensor(list(map(lambda x : x[0], population)), dtype=torch.float32)

lr = 0.001
batch_size = 16

pos_emb_size = 32
pos_emb_vals = torch.empty((problem_size, pos_emb_size))
torch.nn.init.uniform_(pos_emb_vals, -0.01, 0.01)
pos_emb = torch.nn.Parameter(pos_emb_vals)

encoder_layer = torch.nn.TransformerEncoderLayer(d_model=pos_emb_size, nhead=2)
pos_model = torch.nn.TransformerEncoder(encoder_layer, 2)

# pos_model = torch.nn.Sequential(
#     torch.nn.Linear(problem_size, problem_size//2),
#     torch.nn.ReLU(),
#     torch.nn.Linear(problem_size//2, problem_size//4),
#     torch.nn.ReLU(),
#     torch.nn.Linear(problem_size//4, problem_size//2),
#     torch.nn.ReLU(),
#     torch.nn.Linear(problem_size//2, problem_size)
# )

encoder = torch.nn.Sequential(
    torch.nn.Linear(problem_size, 13),
    torch.nn.Tanh(), 
    torch.nn.Linear(13, problem_size)
)
decoder = torch.nn.Sequential(
    torch.nn.Linear(pos_emb_size, problem_size),
    torch.nn.Tanh()
)

optimizer = torch.optim.Adam([{"params": encoder.parameters()}, {"params": decoder.parameters()},
                              {"params":pos_emb}, {"params": pos_model.parameters()}], lr=lr)

epochs = 500
for epoch in range(epochs):
    dataset = DataLoader(TensorDataset(population), batch_size=batch_size, shuffle=True)
    for i,x in enumerate(dataset):
        # shape = (problem_size, pod_emb_size)
        new_pos_embs = pos_model(pos_emb)
        # shape = (16, problem_size)
        latent = encoder(x[0])
        # shape = (16, pos_emb_size)
        latent = latent.matmul(new_pos_embs)
        recon = decoder(latent)
        mse = F.mse_loss(recon, x[0])
        # l1 = 0.00001 * sum(p.abs().sum() for p in model.parameters())
        loss = mse #+ l1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch {}/{} - {}/{} - Loss = {}".format(epoch+1,epochs,i,len(population),mse))

handler.print_statistics(fitnesses)
i = fitnesses.argmax(dim=0)
solution = population[i]
print(fitnesses[i])
latents = encoder[:2](solution)
pos_embs = pos_model(pos_emb)

old_recon = encoder[2:](latents).reshape((1,problem_size)).matmul(pos_embs)
old_recon = torch.sign(decoder(old_recon))

print(latents)

fig, axes = plt.subplots(ncols=4, figsize=(16,4))
for i ,(ax, latent) in enumerate(zip(axes, latents)): 
    l = latents.clone()
    new_fitnesses = []
    values = np.linspace(-1,1,1000)
    for x in values:
        l[i] = x
        new_l = encoder[2:](l)
        new_recon = new_l.reshape((1,problem_size)).matmul(pos_embs)
        new_recon = decoder(new_recon)
        new_recon = torch.sign(new_recon)
        delta_s = new_recon - old_recon
        new_s = torch.sign(solution + delta_s)

        fitness = problem.fitness(new_s.detach().numpy().reshape((problem_size)))
        new_fitnesses.append(fitness)
    ax.plot(values,new_fitnesses)
    
plt.show()
