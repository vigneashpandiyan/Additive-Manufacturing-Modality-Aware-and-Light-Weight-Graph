import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# === Simulated Data ===
np.random.seed(0)
df = pd.DataFrame({
    "data": np.hstack([
        np.random.normal(loc=0.05, scale=0.015, size=1000),
        np.random.normal(loc=0.03, scale=0.010, size=1000),
        np.random.normal(loc=0.04, scale=0.012, size=1000),
        np.random.normal(loc=0.06, scale=0.017, size=1000),
        np.random.normal(loc=0.045, scale=0.014, size=1000),
    ]),
    "label": np.repeat(["70 µm", "80 µm", "90 µm", "100 µm", "110 µm"], 1000)
})

colors = {
    "70 µm": "#A3E4A1",
    "80 µm": "#F6DD9A",
    "90 µm": "#FDAE6B",
    "100 µm": "#FD7F6B",
    "110 µm": "#FF4C4C"
}

labels = df['label'].unique()
n = len(labels)
n_cols = 2
n_rows = (n + 1) // n_cols  # e.g., 5 → 3 rows

fig = plt.figure(figsize=(10, 12))
gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
axes = []

for i, label in enumerate(labels):
    sub_df = df[df['label'] == label]

    if i < 4:
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
    else:
        # Add last subplot across both columns, then shrink & center it
        ax = fig.add_subplot(gs[-1, :])
        box = ax.get_position()
        full_width = box.width
        desired_width = full_width * 0.45  # match normal subplots
        x_center = box.x0 + box.width / 2
        new_x0 = x_center - desired_width / 2
        ax.set_position([new_x0, box.y0, desired_width, box.height])

    sns.kdeplot(
        data=sub_df,
        x='data',
        ax=ax,
        fill=True,
        linewidth=2,
        color=colors[label],
        label=label
    )

    ax.set_title(f"Layer thickness: {label}", fontsize=14)
    ax.set_xlabel("Data spread", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(loc='upper right')
    ax.tick_params(labelsize=12)

plt.tight_layout()
plt.show()
