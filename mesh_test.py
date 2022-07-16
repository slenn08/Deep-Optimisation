import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import sys
sys.path.append(".")
from COProblems.OptimisationProblem import ECProblem
from OptimAE import OptimAEHandler

def get_distances(structure_points: torch.Tensor, solution_points: torch.Tensor):
    """
    
    Args:
        structure_points: torch.Tensor
            shape = (num_points, points_dim)
        solution_points: torch.Tensor
            shape = (pop_size, points_dim)
    Returns:
        Tensor of shape = (pop_size, num_points) showing the distances from a solution
        point to every structure point.
    """
    return torch.cdist(solution_points, structure_points)

def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
            shape = (Nb, D)
        y: torch/Tensor
            shape = (Np, D)
    """
    x_sizes = torch.sqrt(torch.sum(torch.pow(x,2),dim=1))
    y_sizes = torch.sqrt(torch.sum(torch.pow(y,2),dim=1))
    return x.matmul(y.T) / torch.outer(x_sizes, y_sizes)

problem_size = 256
compression = "nov"
environment = "gc"
pop_size = 128
problem = ECProblem(problem_size,compression,environment)

handler = OptimAEHandler(None, problem)
population, fitnesses = handler.generate_population(pop_size)
population, fitnesses, _, _ = handler.hillclimb(population, fitnesses, 128)

# population = data["nov_gc_256"][:pop_size]
# fitnesses = torch.tensor(list(map(lambda x : x[1], population)), dtype=torch.float32)
# population = torch.tensor(list(map(lambda x : x[0], population)), dtype=torch.float32)

lr = 0.001
batch_size = 16

points_dim = 16
points_number = problem_size
pos_emb_vals = torch.empty((points_number, 1, points_dim))
torch.nn.init.uniform_(pos_emb_vals, -0.01, 0.01)
pos_emb = torch.nn.Parameter(pos_emb_vals)

encoder_layer = torch.nn.TransformerEncoderLayer(d_model=points_dim, nhead=2)
pos_model = torch.nn.TransformerEncoder(encoder_layer, 2)

encoder = torch.nn.Sequential(
    torch.nn.Linear(problem_size, 4),
    torch.nn.Tanh(), 
    torch.nn.Linear(4, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, points_dim)
)
decoder = torch.nn.Sequential(
    torch.nn.Linear(points_number, problem_size),
    torch.nn.Tanh()
)

optimizer = torch.optim.Adam([{"params": encoder.parameters()}, {"params": decoder.parameters()},
                              {"params":pos_emb}, {"params": pos_model.parameters()}], lr=lr)

# TODO try dot product between pos embedding and solution embedding instead of cdist
epochs = 500
for epoch in range(epochs):
    dataset = DataLoader(TensorDataset(population), batch_size=batch_size, shuffle=True)
    for i,x in enumerate(dataset):
        new_pos_embs = pos_model(pos_emb)
        sol_emb = encoder(x[0])
        # print(sol_emb.shape)
        # print(new_pos_embs.T.shape)
        #print(sol_emb.matmul(new_pos_embs.T[:,0,:]))
        distances = torch.cdist(sol_emb, new_pos_embs[:,0,:])
        #distances = cosine_similarity(sol_emb, new_pos_embs[:,0,:])
        recon = decoder(distances)
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
latents = encoder[:2](solution)
pos_embs = pos_model(pos_emb)[:,0,:]

sol_embs = encoder[2:](latents).reshape((1,points_dim))
distances = torch.cdist(sol_embs, pos_embs)
old_recon = torch.sign(decoder(distances)).reshape((problem_size))

print(latents)

fig, axes = plt.subplots(ncols=4, figsize=(16,4))
for i ,(ax, latent) in enumerate(zip(axes, latents)): 
    l = latents.clone()
    new_fitnesses = []
    values = np.linspace(-1,1,1000)
    for x in values:
        l[i] = x
        new_sol_embs = encoder[2:](l).reshape((1,points_dim))
        distances = torch.cdist(new_sol_embs, pos_embs)
        #distances = cosine_similarity(new_sol_embs, pos_embs)
        new_recon = decoder(distances)
        new_recon = torch.sign(new_recon).reshape((problem_size))
        delta_s = new_recon - old_recon
        new_s = torch.sign(solution + delta_s)

        fitness = problem.fitness(new_s.detach().numpy().reshape((problem_size)))
        new_fitnesses.append(fitness)
    ax.plot(values,new_fitnesses)
    
plt.show()
