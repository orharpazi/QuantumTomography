from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

THRESHOLD_FACTOR = 0.6406533066132265
plt.style.use(['ieee'])

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "data" / "processed"
files = list(INPUT_DIR.glob("*.csv"))


# Plot each CSV into its own subplot (4x4 grid)
fig, axes = plt.subplots(4, 4, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()
i = -1
for i, file in enumerate(sorted(files)):
    data = np.genfromtxt(file, delimiter=",", skip_header=1)
    #normalise the dataset
    data[:,0] = (data[:,0] - data[:,0].min()) / (data[:,0].max() - data[:,0].min()) if data[:,0].max() != data[:,0].min() else 0
    data[:, 1] = (data[:, 1] - data[:, 1].min()) / (data[:, 1].max() - data[:, 1].min()) if data[:, 1].max() != data[:, 1].min() else 0
    data[:, 2] = (data[:, 2] - data[:, 2].min()) / (data[:, 2].max() - data[:, 2].min()) if data[:, 2].max() != data[:, 2].min() else 0
    if data.size == 0:
        continue
    timestamps = data[:, 0]
    webcam_1_avg = data[:, 1]
    webcam_2_avg = data[:, 2]

    ax = axes[i]
    ax.plot(timestamps, webcam_1_avg, label="Webcam 1")
    ax.plot(timestamps, webcam_2_avg, label="Webcam 2")
    ax.axhline(y=THRESHOLD_FACTOR, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=THRESHOLD_FACTOR, color='g', linestyle='--', alpha=0.5)
    alpha, beta = [val / 10.0 if val > 113 else val for val in map(float, file.stem.split("_")[0].split(","))]    
    ax.set_title(r'$\alpha = {}^\circ, \beta = {}^\circ$'.format(alpha, beta))
    ax.legend(fontsize="small", loc="upper left")

# Turn off any unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

fig.suptitle("Detector Intensities at Different Angle Pairs, Normalised", fontsize=21)

fig.tight_layout()
plt.savefig(ROOT_DIR / "data" / "all_intensities.png", dpi=300)