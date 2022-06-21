import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import Isomap,LocallyLinearEmbedding, TSNE
import matplotlib.pyplot as plt

import sys

sys.path.append(".")
from Models.DOVAE2 import *

def cmyk_to_rgb(c, m, y, k, cmyk_scale=1.0, rgb_scale=1.0):
    r = rgb_scale * (1.0 - c / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    g = rgb_scale * (1.0 - m / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    b = rgb_scale * (1.0 - y / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    return [r, g, b]

nov_bbs = [[-1,-1,-1,-1], [-1,1,-1,1], [1,-1,1,-1], [1,1,1,1]]
ov_bbs = [[-1,-1,-1,-1], [-1,1,-1,1], [1,-1,-1,1], [1,1,1,1]]
ndov_bbs = [[-1,-1,-1,1], [-1,-1,1,-1], [-1,1,-1,-1], [1,-1,-1,-1]]
npov_bbs = [[1,1,1,1], [-1,-1,1,1], [-1,1,-1,-1], [1,-1,-1,-1]]

    
# Hyper-parameters
change_tolerance = 512
pop_size = 256
batch_size = 16
window_size = 32
lr = 0.001
kl_weight = 0.1
compression = 0.8
no_change_prob = 0.9
problem_size = 32
problem = ECProblem(problem_size, "npov", "hgc")
problem.max_fitness = float('inf')
latent_size = int(compression*problem_size)

model = DeepOptimizer(problem_size, latent_size)
model.to(device)
population = [problem.random_solution() for _ in range(pop_size)]
# Convert to pytorch tensor, attach fitness to data
population = list(map(lambda x : (torch.tensor(x, dtype=torch.float32, device=device), problem.fitness(x)), population))

print("Optimising")
population = optimize_population_hillclimb(population, problem, window_size, pop_size)
print("Learning")
model.reset_weights()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
learn_from_population(model, population, optimizer, batch_size, latent_size, kl_weight)
#####
population, total_evaluations = optimize_population_model(population, model, problem,
                                                          0,3.0, latent_size,
                                                          no_change_prob,change_tolerance,window_size,
                                                          pop_size)

model.reset_weights()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
learn_from_population(model, population, optimizer, batch_size, latent_size, kl_weight)
#####

print("Plotting")
fitnesses = list(map(lambda x : x[1], population))
max_fitness = max(fitnesses)
normalised_f = np.array(fitnesses)/max_fitness
stacked_pop = torch.stack(list(map(lambda x : x[0], population)))
latents,_ = model.encode(stacked_pop)
latents = latents.detach()

bb_counts = []
for latent in latents:
    counts = [0,0,0,0]
    latent_list = to_int_list(latent)
    for i, bb in enumerate(npov_bbs):
        for latent_bb in [latent_list[i:i + 4] for i in range(0, len(latent_list), 4)]:
            if latent_bb == bb:
                counts[i] += 1
    bb_counts.append(counts)

bb_counts = np.array(bb_counts, dtype=np.float64)
bb_counts /= bb_counts.max()
bb_colours = np.zeros((bb_counts.shape[0],3))
for i,bb_count in enumerate(bb_counts):
    bb_colours[i] = cmyk_to_rgb(*bb_count)
latents = latents.detach().numpy()

embedding = LocallyLinearEmbedding(
        n_neighbors=3, n_components=2, method="standard", eigen_solver="dense"
    )
# embedding = TruncatedSVD(n_components=2)
projection = embedding.fit_transform(latents)

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
# fig = plt.figure()
# axs = fig.add_subplot(projection='3d')
# for ax in axs:
axs.scatter(projection[:,0], projection[:,1], s=30, c=bb_colours, alpha=normalised_f)
plt.show()

