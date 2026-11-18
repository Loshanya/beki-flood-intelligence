
import os
import numpy as np
import matplotlib.pyplot as plt

from joblib import load
from matplotlib.colors import BoundaryNorm

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "ML")
MODEL_DIR = os.path.join(DATA, "models")
OUTPUT_DIR = os.path.join(DATA, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

Q = float(input("Enter discharge (200-10000 cumecs): "))

if not 200 <= Q <= 10000:
    raise ValueError("Discharge must be between 200 and 10000.")

print("\nLoading models...")

wet_bundle = load(
    os.path.join(MODEL_DIR, "wet_classifier.joblib")
)
wet_model = wet_bundle["model"]
threshold = wet_bundle["threshold"]

depth_model = load(
    os.path.join(MODEL_DIR, "depth_rf.joblib")
)
velocity_model = load(
    os.path.join(MODEL_DIR, "velocity_rf.joblib")
)

# Grid dimensions
rows, cols = 286, 600
row_grid, col_grid = np.indices((rows, cols))

features = np.column_stack([
    np.full(rows * cols, Q),
    row_grid.ravel(),
    col_grid.ravel()
])

print("Predicting wet/dry probability...")

wet_probability = wet_model.predict_proba(features)[:, 1]
wet_mask = wet_probability >= threshold

print("Predicting water depth...")
depth = depth_model.predict(features)

print("Predicting velocity...")
velocity = velocity_model.predict(features)

depth = np.maximum(depth, 0)
velocity = np.maximum(velocity, 0)

# Apply wet/dry classification
depth = depth.reshape(rows, cols)
velocity = velocity.reshape(rows, cols)
wet_mask = wet_mask.reshape(rows, cols)

depth[~wet_mask] = 0.1
velocity[~wet_mask] = 0.0

# FHI and original hazard thresholds
fhi = depth * (velocity + 0.5)

classes = np.zeros((rows, cols), dtype=np.uint8)

classes[wet_mask & (fhi <= 0.5)] = 1
classes[wet_mask & (fhi > 0.5) & (fhi <= 1)] = 2
classes[wet_mask & (fhi > 1) & (fhi <= 2)] = 3
classes[wet_mask & (fhi > 2) & (fhi <= 3)] = 4
classes[wet_mask & (fhi > 3)] = 5

print("\n========== ML PREDICTION V2 ==========")
print("Discharge:", Q)
print("Wet cells:", int(wet_mask.sum()))
print("Dry cells:", int((~wet_mask).sum()))
print("Mean wet depth:", round(depth[wet_mask].mean(), 4))
print("Maximum depth:", round(depth.max(), 4))
print("Mean wet velocity:", round(velocity[wet_mask].mean(), 4))
print("Maximum velocity:", round(velocity.max(), 4))

print("\nHazard class distribution:")
for c in range(6):
    print(f"Class {c}: {int((classes == c).sum())} cells")

# Save prediction
q_label = str(int(Q)) if Q.is_integer() else str(Q).replace(".", "_")

np.savez_compressed(
    os.path.join(OUTPUT_DIR, f"ml_prediction_v2_Q{q_label}.npz"),
    discharge=Q,
    depth=depth,
    velocity=velocity,
    wet_probability=wet_probability.reshape(rows, cols),
    wet_mask=wet_mask,
    fhi=fhi,
    classes=classes
)

# Load projected coordinate grids
coords = np.load(
    os.path.join(ROOT, "deployment", "coordinate_grids.npz")
)

X = coords["x"]
Y = coords["y"]

if X.shape != (rows + 1, cols + 1) or Y.shape != X.shape:
    raise ValueError("Coordinate grid dimensions do not match prediction.")

# Plot hazard map
fig, ax = plt.subplots(figsize=(10, 12))

cmap = plt.get_cmap("RdYlGn_r", 6)
norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)

mesh = ax.pcolormesh(
    X, Y, classes,
    cmap=cmap,
    norm=norm,
    shading="flat"
)

cbar = fig.colorbar(
    mesh, ax=ax,
    ticks=range(6),
    boundaries=np.arange(-0.5, 6.5, 1)
)

cbar.ax.set_yticklabels([
    "0 - Dry",
    "1 - Very Low",
    "2 - Low",
    "3 - Moderate",
    "4 - High",
    "5 - Very High"
])

ax.set_title(
    f"ML Flood Hazard Prediction - Q={Q:g} cumecs"
)
ax.set_xlabel("Projected X coordinate")
ax.set_ylabel("Projected Y coordinate")
ax.set_aspect("equal")

plt.tight_layout()

map_path = os.path.join(
    OUTPUT_DIR, f"ml_hazard_map_v2_Q{q_label}.png"
)

plt.savefig(map_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nPrediction saved successfully.")
print("Map:", map_path)
