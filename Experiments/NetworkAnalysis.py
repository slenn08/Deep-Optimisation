import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import itertools
import sys
import torch
from torch import nn
from scipy.stats import norm

sys.path.append(".")
from Models import DOVAE2 as VDO
from Models import DO2 as DO

batch_size = 16
window_size = 32
lr = 0.001
kl_weight = 0.1
model_VDO = VDO.DeepOptimizer(4,2)
model_VDO.reset_weights()
optimizer = torch.optim.Adam(model_VDO.parameters(), lr=lr)

NDOV_dataset = torch.tensor([[1,-1,-1,-1],[-1,1,-1,-1],[-1,-1,1,-1],[-1,-1,-1,1]], dtype=torch.float32)
NPOV_dataset = torch.tensor([[1,1,1,1],[-1,-1,1,1],[-1,1,-1,-1],[1,-1,-1,-1]], dtype=torch.float32)
NOV_dataset = torch.tensor([[1,1,1,1],[-1,1,-1,1],[1,-1,1,-1],[-1,-1,-1,-1]], dtype=torch.float32)
OV_dataset = torch.tensor([[1,1,1,1],[-1,1,-1,1],[1,-1,-1,1],[-1,-1,-1,-1]], dtype=torch.float32)

# dataset = OV_dataset

# print("VDO Training")
# for _ in range(20000):
#     VDO.learn_from_sample(model_VDO, dataset, optimizer, kl_weight)

# output, mu, logvar, _ = model_VDO(dataset)
# output = model_VDO.decode(mu)
# mu = mu.detach()
# logvar = logvar.detach()
# output = output.detach().numpy()

# fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(4,2))
# ax1.imshow(dataset.detach().numpy(), cmap="Greys", interpolation="none")
# ax1.set_title("OV Input")
# ax1.axes.yaxis.set_visible(False)
# ax1.axes.xaxis.set_visible(False)
# img2 = ax2.imshow(output, cmap='Greys', interpolation='none')
# ax2.set_title("OV Output")
# ax2.axes.yaxis.set_visible(False)
# ax2.axes.xaxis.set_visible(False)
# divider = make_axes_locatable(ax2)
# cax2 = divider.append_axes("right", size="5%", pad=0.05)
# cbar = fig.colorbar(img2, cax=cax2)
# ax2.set_aspect("auto")
# ax1.set_aspect("auto")
# plt.show()
# print(output)
# print(mu.numpy())
# print(torch.exp(0.5*logvar).numpy())

# std = torch.exp(0.5*logvar).detach()

# fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(8,4))
# for m,s,c,ps in zip(mu.numpy(), std.numpy(),["b","g","y","m"], dataset):
#     mu1,mu2 = m
#     std1,std2 = s
#     x1 = np.linspace(mu1-std1*3, mu1+std1*3, 200)
#     y1 = norm.pdf(x1, mu1, std1)
#     x2 = np.linspace(mu2-std2*3, mu2+std2*3, 200)
#     y2 = norm.pdf(x2, mu2, std2)
#     ps = (ps+1)/2
#     ps = ps.tolist()
#     ps = [int(i) for i in ps]
#     ax1.plot(x1,y1,c=c,label=str(ps))
#     ax2.plot(x2,y2,c=c,label=str(ps))
# ax1.legend()
# ax2.legend()
# ax1.set_title("E=NDOV")
# ax2.set_xlabel("Activation Values")
# ax1.set_ylabel("Hidden Node 1")
# ax2.set_ylabel("Hidden Node 2")
# ax1.axes.yaxis.set_ticks([])
# ax2.axes.yaxis.set_ticks([])
# plt.show()
    
dataset = NPOV_dataset
print("DO Training")
model_DO = DO.DeepOptimizer(4)
model_DO.add_layer(2)
optimizer = torch.optim.Adam(model_DO.parameters(), lr=lr)
criterion = nn.MSELoss()
for _ in range(10000):
    DO.learn_from_sample(model_DO, dataset, criterion, optimizer, 0.0)

output, z = model_DO(dataset)
z = z.detach().numpy()
for z_var, c, ps in zip(z,["b","g","y","m"],dataset):
    ps = (ps+1)/2
    ps = ps.tolist()
    ps = [int(i) for i in ps]
    plt.scatter(1,z_var[0],c=c,alpha=0.5,label=str(ps),s=120)
    plt.scatter(2,z_var[1],c=c,alpha=0.5,s=120)
plt.xticks([1,2])
plt.xlabel("Hidden Node")
plt.ylabel("Activation")
plt.title("E=NPOV")
plt.legend()
plt.show()

output = output.detach().numpy()
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(4,2))
ax1.imshow(dataset.detach().numpy(), cmap="Greys", interpolation="none")
ax1.set_title("NPOV Input")
ax1.axes.yaxis.set_visible(False)
ax1.axes.xaxis.set_visible(False)
img2 = ax2.imshow(output, cmap='Greys', interpolation='none')
ax2.set_title("NPOV Output")
ax2.axes.yaxis.set_visible(False)
ax2.axes.xaxis.set_visible(False)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(img2, cax=cax2)
ax2.set_aspect("auto")
ax1.set_aspect("auto")
plt.show()