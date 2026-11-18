
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "ML", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "ML", "outputs")
DATA_DIR = os.path.join(BASE_DIR, "deployment")

os.makedirs(OUTPUT_DIR, exist_ok=True)

Q = 5350  # Change this to any discharge between 200 and 10000

ROWS = 286
COLS = 600

# ==========================================
# LOAD MODELS
# ==========================================

print("Loading trained models...")

wet_data = joblib.load(
    os.path.join(MODEL_DIR, "wet_classifier.joblib")
)

wet_model = wet_data["model"]
wet_threshold = wet_data["threshold"]

depth_model = joblib.load(
    os.path.join(MODEL_DIR, "depth_rf_log.joblib")
)

velocity_model = joblib.load(
    os.path.join(MODEL_DIR, "velocity_rf_log.joblib")
)

print("All models loaded successfully.")

# ==========================================
# LOAD COORDINATES
# ==========================================

coord_file = os.path.join(DATA_DIR, "coordinate_grids.npz")
coords = np.load(coord_file)

X = coords["x"]
Y = coords["y"]

if X.shape != (ROWS + 1, COLS + 1):
    raise ValueError(f"Unexpected X coordinate shape: {X.shape}")

if Y.shape != (ROWS + 1, COLS + 1):
    raise ValueError(f"Unexpected Y coordinate shape: {Y.shape}")

# ==========================================
# CREATE GRID FEATURES
# ==========================================

print(f"\nGenerating predictions for Q = {Q} cumecs...")

row_grid, col_grid = np.indices((ROWS, COLS))

features = np.column_stack([
    np.full(ROWS * COLS, Q),
    row_grid.ravel(),
    col_grid.ravel()
])

# ==========================================
# PREDICT WET / DRY
# ==========================================

print("Predicting wet/dry cells...")

wet_probabilities = wet_model.predict_proba(features)[:, 1]

wet_mask = wet_probabilities >= wet_threshold

# ==========================================
# PREDICT DEPTH AND VELOCITY
# ==========================================

print("Predicting depth...")

depth_log = depth_model.predict(features)
depth_all = np.expm1(depth_log)

print("Predicting velocity...")

velocity_log = velocity_model.predict(features)
velocity_all = np.expm1(velocity_log)

# ==========================================
# APPLY WET / DRY MASK
# ==========================================

depth = np.zeros(ROWS * COLS)
velocity = np.zeros(ROWS * COLS)

depth[wet_mask] = np.maximum(depth_all[wet_mask], 0.1)
velocity[wet_mask] = np.maximum(velocity_all[wet_mask], 0)

depth = depth.reshape(ROWS, COLS)
velocity = velocity.reshape(ROWS, COLS)

# ==========================================
# FLOOD HAZARD INDEX
# ==========================================

FHI = depth * (velocity + 0.5)

hazard = np.zeros((ROWS, COLS), dtype=np.int8)

wet = depth > 0.1

hazard[wet & (FHI <= 0.5)] = 1
hazard[wet & (FHI > 0.5) & (FHI <= 1)] = 2
hazard[wet & (FHI > 1) & (FHI <= 2)] = 3
hazard[wet & (FHI > 2) & (FHI <= 3)] = 4
hazard[wet & (FHI > 3)] = 5

# ==========================================
# SAVE PREDICTIONS
# ==========================================

output_file = os.path.join(
    OUTPUT_DIR,
    f"ml_prediction_v3_Q{Q}.npz"
)

np.savez_compressed(
    output_file,
    discharge=Q,
    depth=depth,
    velocity=velocity,
    FHI=FHI,
    hazard=hazard,
    wet_mask=wet
)

print("\nPrediction saved:", output_file)

# ==========================================
# PRINT SUMMARY
# ==========================================

print("\n========== V3 PREDICTION SUMMARY ==========")

print("Discharge:", Q, "cumecs")
print("Wet cells:", np.sum(wet))
print("Dry cells:", np.sum(~wet))

print("Mean depth:", round(float(depth[wet].mean()), 4))
print("Maximum depth:", round(float(depth.max()), 4))

print("Mean velocity:", round(float(velocity[wet].mean()), 4))
print("Maximum velocity:", round(float(velocity.max()), 4))

print("\nHazard class distribution:")

for cls in range(6):
    count = np.sum(hazard == cls)
    percentage = count / hazard.size * 100
    print(
        f"Class {cls}: {count} cells ({percentage:.2f}%)"
    )

# ==========================================
# PLOT HAZARD MAP
# ==========================================

print("\nGenerating hazard map...")

plt.figure(figsize=(12, 8))

mesh = plt.pcolormesh(
    X,
    Y,
    hazard,
    shading="flat",
    cmap="RdYlGn_r",
    vmin=0,
    vmax=5
)

plt.colorbar(
    mesh,
    ticks=[0, 1, 2, 3, 4, 5],
    label="Flood Hazard Class"
)

plt.title(f"Beki River ML Flood Hazard Map - Q={Q} cumecs")
plt.xlabel("Projected X")
plt.ylabel("Projected Y")
plt.axis("equal")
plt.tight_layout()

map_file = os.path.join(
    OUTPUT_DIR,
    f"ml_hazard_map_v3_Q{Q}.png"
)

plt.savefig(map_file, dpi=300)
plt.show()

print("Map saved:", map_file)
print("\nV3 prediction completed successfully.")
