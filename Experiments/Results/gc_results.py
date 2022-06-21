import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import numpy as np

sizes = np.array([16,32,64,128,256])

nov_ae = np.array([40,524.7,2254.0,1560.1,22252.67])
ov_ae = np.array([648.6,3556.7,11883.2,36930.4,97212.33])
ndov_ae = np.array([1074.9,10556.9,33722,123094.3,435383.67])
npov_ae = np.array([2639.1,45647.6,489662.33,1043856,12734823])

nov_vae = np.array([171.2,672.8,3288.1,8206.66,17600.33])
ov_vae = np.array([570.8,3844.7,10917.2,54798.0,211970.66])
ndov_vae = np.array([242.7,12531,33192.66,211403,923013])
npov_vae = np.array([1109.0,23511,123500.4,602831.2,3370243])

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,6))

axes[0,0].scatter(sizes, nov_ae, marker="o",label="AE")
axes[0,0].scatter(sizes, nov_vae, marker="x",label="VAE")
axes[0,0].set_title("E=GC, C=nOV")

axes[0,1].scatter(sizes, ov_ae, marker="o",label="AE")
axes[0,1].scatter(sizes, ov_vae, marker="x",label="VAE")
axes[0,1].set_title("E=GC, C=OV")

axes[1,0].scatter(sizes, ndov_ae, marker="o",label="AE")
axes[1,0].scatter(sizes, ndov_vae, marker="x",label="VAE")
axes[1,0].set_title("E=GC, C=nDOV")

axes[1,1].scatter(sizes, npov_ae, marker="o",label="AE")
axes[1,1].scatter(sizes, npov_vae, marker="x",label="VAE")
axes[1,1].set_title("E=GC, C=nPOV")

for ax in axes.reshape(-1):
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(ticks=sizes, labels=sizes)
    ax.minorticks_off()
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Evaluations")
    ax.legend()

fig.tight_layout()

plt.show()
