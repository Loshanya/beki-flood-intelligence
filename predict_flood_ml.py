
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Project paths
BASE = Path(__file__).resolve().parent
ML_ROOT = BASE / "ML"
MODEL_DIR = ML_ROOT / "models"
OUTPUT_DIR = ML_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load trained models
depth_model = joblib.load(MODEL_DIR / "depth_rf.joblib")
velocity_model = joblib.load(MODEL_DIR / "velocity_rf.joblib")

# User discharge
Q = float(input("Enter discharge in cumecs (200-10000): "))

if not 200 <= Q <= 10000:
    raise ValueError("Discharge must be between 200 and 10000.")

# Grid dimensions
ROWS = 286
COLS = 600

print("\nGenerating spatial prediction features...")

row, col = np.indices((ROWS, COLS))

X = np.column_stack([
    np.full(ROWS * COLS, Q),
    row.ravel(),
    col.ravel()
])

print("Predicting water depth...")
depth = depth_model.predict(X).reshape(ROWS, COLS)

print("Predicting velocity...")
velocity = velocity_model.predict(X).reshape(ROWS, COLS)

# Apply physically basic constraints
depth = np.maximum(depth, 0)
velocity = np.maximum(velocity, 0)

# Dry cells have zero velocity
dry = depth <= 0.1
velocity[dry] = 0

# Calculate FHI
fhi = depth * (velocity + 0.5)

# Hazard classification
classes = np.zeros((ROWS, COLS), dtype=np.int8)
wet = depth > 0.1

classes[wet & (fhi <= 0.5)] = 1
classes[wet & (fhi > 0.5) & (fhi <= 1)] = 2
classes[wet & (fhi > 1) & (fhi <= 2)] = 3
classes[wet & (fhi > 2) & (fhi <= 3)] = 4
classes[wet & (fhi > 3)] = 5

# Save prediction grids
np.savez_compressed(
    OUTPUT_DIR / f"ml_prediction_Q{int(Q)}.npz",
    discharge=Q,
    depth=depth,
    velocity=velocity,
    fhi=fhi,
    classes=classes
)

# Load projected coordinates if available
coordinate_file = BASE / "deployment" / "coordinate_grids.npz"

if coordinate_file.exists():
    coords = np.load(coordinate_file)

    print("Available coordinate keys:", coords.files)

    X_grid = None
    Y_grid = None

    for key in coords.files:
        arr = coords[key]

        if arr.shape == (287, 601):
            if X_grid is None:
                X_grid = arr
            elif Y_grid is None:
                Y_grid = arr

    if X_grid is None or Y_grid is None:
        print("Coordinate grids unavailable. Using grid indices.")
        X_grid = None
        Y_grid = None
else:
    X_grid = None
    Y_grid = None

# Generate hazard map
plt.figure(figsize=(11, 8))

if X_grid is not None and Y_grid is not None:
    plt.pcolormesh(
        X_grid,
        Y_grid,
        classes,
        shading="flat",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=5
    )
    plt.xlabel("Projected X coordinate")
    plt.ylabel("Projected Y coordinate")
    plt.gca().set_aspect("equal")
else:
    plt.imshow(
        classes,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=5,
        origin="upper"
    )
    plt.xlabel("Grid column")
    plt.ylabel("Grid row")

plt.colorbar(
    ticks=[0, 1, 2, 3, 4, 5],
    label="Hazard class"
)

plt.title(f"ML-Based Flood Hazard Prediction - Q={Q:.0f} cumecs")
plt.tight_layout()

map_file = OUTPUT_DIR / f"ml_hazard_map_Q{int(Q)}.png"
plt.savefig(map_file, dpi=300)
plt.close()

print("\n========== ML PREDICTION SUMMARY ==========")
print("Discharge:", Q)
print("Mean predicted depth:", round(float(depth.mean()), 4))
print("Maximum predicted depth:", round(float(depth.max()), 4))
print("Mean predicted velocity:", round(float(velocity.mean()), 4))
print("Maximum predicted velocity:", round(float(velocity.max()), 4))
print("Wet cells:", int(wet.sum()))

print("\nHazard class distribution:")

for c in range(6):
    count = int(np.sum(classes == c))
    print(f"Class {c}: {count} cells")

print("\nPrediction grids saved successfully.")
print("Map saved at:", map_file)
