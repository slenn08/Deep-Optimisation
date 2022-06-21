import matplotlib.pyplot as plt
import numpy as np

best = np.array([11,23,47,95,191])

nov_ae = np.array([11,23,47,95,191]) / best
nov_vae = np.array([11,23,47,95,191]) / best
ov_ae = np.array([11,23,47,83,156]) / best
ov_vae = np.array([11,23,47,92,173]) / best
ndov_ae = np.array([11,23,45,82,171]) / best
ndov_vae = np.array([11,23,47,89,164]) / best
npov_ae = np.array([11,23,43,65,121]) / best
npov_vae = np.array([11,23,44,77,150]) / best

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,6))
labels = [16, 32, 64, 128, 256]
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

rects1 = axes[0,0].bar(x - 0.5*width, nov_ae, width, label='AE')
rects2 = axes[0,0].bar(x + 0.5*width, nov_vae, width, label='VAE')
axes[0,0].set_title("E=nOV")
# axes[0,0].bar_label(rects1, padding=3)
# axes[0,0].bar_label(rects2, padding=3)

rects1 = axes[0,1].bar(x - 0.5*width, ov_ae, width, label='AE')
rects2 = axes[0,1].bar(x + 0.5*width, ov_vae, width, label='VAE')
axes[0,1].set_title("E=OV")
# axes[0,1].bar_label(rects1, padding=3)
# axes[0,1].bar_label(rects2, padding=3)

rects1 = axes[1,0].bar(x - 0.5*width, ndov_ae, width, label='AE')
rects2 = axes[1,0].bar(x + 0.5*width, ndov_vae, width, label='VAE')
axes[1,0].set_title("E=nDOV")
# axes[1,0].bar_label(rects1, padding=3)
# axes[1,0].bar_label(rects2, padding=3)

rects1 = axes[1,1].bar(x - 0.5*width, npov_ae, width, label='AE')
rects2 = axes[1,1].bar(x + 0.5*width, npov_vae, width, label='VAE')
axes[1,1].set_title("E=nPOV")
# axes[1,1].bar_label(rects1, padding=3)
# axes[1,1].bar_label(rects2, padding=3)

for ax in axes.reshape(-1):
    ax.set_xticks(x, labels)
    ax.set(ylim=[0.6, 1.01])
    ax.legend(loc="lower left")
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Normalised Fitness")
fig.tight_layout()
plt.show()