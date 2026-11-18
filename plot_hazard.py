from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

folder = Path("outputs/Q_300")
hazard = pd.read_csv(
    folder / "hazard_class_300.csv", header=None
).to_numpy()

# Class 0 is shown as unclassified; classes 1–5 are hazard levels
colors = [
    "#d9d9d9",  # 0: dry/unclassified
    "#2c7bb6",  # 1: very low
    "#abd9e9",  # 2: low
    "#ffffbf",  # 3: moderate
    "#fdae61",  # 4: high
    "#d7191c",  # 5: very high
]

cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)

fig, ax = plt.subplots(figsize=(14, 6))
ax.imshow(hazard, cmap=cmap, norm=norm, interpolation="nearest")
ax.set_title("Preliminary Flood Hazard Classes — 300 cumecs")
ax.set_xlabel("Grid column (not geographic longitude)")
ax.set_ylabel("Grid row (not geographic latitude)")

labels = [
    "Dry / unclassified",
    "Very low",
    "Low",
    "Moderate",
    "High",
    "Very high",
]
legend = [
    Patch(facecolor=colors[i], label=labels[i])
    for i in range(6)
]
ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1))
plt.tight_layout()

output = folder / "hazard_map_300_grid.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
plt.show()

print(f"Map saved to: {output.resolve()}")