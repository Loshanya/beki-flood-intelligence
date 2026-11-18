
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    accuracy_score
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "ML")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")

Q = 5350
Q_LOW = 5200
Q_HIGH = 5400

# --------------------------------------------------
# 1. Locate hydraulic scenario folders and files
# --------------------------------------------------

def find_folder(q):
    matches = glob.glob(
        os.path.join(DATA_DIR, f"BEKI_FinalQ_{q}_GVF*")
    )
    if not matches:
        raise FileNotFoundError(f"Scenario folder not found: Q={q}")
    return matches[0]


def find_file(folder, prefix):
    matches = glob.glob(os.path.join(folder, prefix + "*.xlsx"))
    if not matches:
        raise FileNotFoundError(
            f"{prefix} Excel file not found in {folder}"
        )
    return matches[0]


def read_grid(folder, prefix):
    path = find_file(folder, prefix)
    print("Reading:", path)
    return pd.read_excel(path, header=None).to_numpy(dtype=float)


# --------------------------------------------------
# 2. Load hydraulic grids
# --------------------------------------------------

print("\nLoading hydraulic scenarios...")

folder_low = find_folder(Q_LOW)
folder_high = find_folder(Q_HIGH)

h_low = read_grid(folder_low, "h_avg")
h_high = read_grid(folder_high, "h_avg")

v_low = read_grid(folder_low, "rv_avg")
v_high = read_grid(folder_high, "rv_avg")

if not (
    h_low.shape == h_high.shape ==
    v_low.shape == v_high.shape
):
    raise ValueError("Hydraulic grid dimensions do not match.")

# Linear interpolation between 5200 and 5400
weight = (Q - Q_LOW) / (Q_HIGH - Q_LOW)

h_hyd = h_low * (1 - weight) + h_high * weight
v_hyd = v_low * (1 - weight) + v_high * weight

print("Hydraulic interpolation completed.")
print("Hydraulic grid shape:", h_hyd.shape)


# --------------------------------------------------
# 3. Load ML predictions
# --------------------------------------------------

ml_path = os.path.join(
    OUTPUT_DIR, f"ml_prediction_Q{Q}.npz"
)

ml = np.load(ml_path)

h_ml = ml["depth"]
v_ml = ml["velocity"]
fhi_ml = ml["fhi"]
class_ml = ml["classes"]

if h_ml.shape != h_hyd.shape:
    raise ValueError(
        f"Grid mismatch: ML {h_ml.shape}, hydraulic {h_hyd.shape}"
    )


# --------------------------------------------------
# 4. Calculate hydraulic FHI and hazard classes
# --------------------------------------------------

def calculate_fhi_and_classes(depth, velocity):
    fhi = depth * (velocity + 0.5)

    classes = np.zeros(depth.shape, dtype=np.uint8)

    wet = depth > 0.1

    classes[wet & (fhi <= 0.5)] = 1
    classes[wet & (fhi > 0.5) & (fhi <= 1)] = 2
    classes[wet & (fhi > 1) & (fhi <= 2)] = 3
    classes[wet & (fhi > 2) & (fhi <= 3)] = 4
    classes[wet & (fhi > 3)] = 5

    return fhi, classes


fhi_hyd, class_hyd = calculate_fhi_and_classes(
    h_hyd, v_hyd
)


# --------------------------------------------------
# 5. Calculate performance metrics
# --------------------------------------------------

def calculate_metrics(name, actual, predicted):
    actual = actual.ravel()
    predicted = predicted.ravel()

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print("-" * 45)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    return mae, rmse, r2


print("\n========== DEPTH COMPARISON ==========")
depth_metrics = calculate_metrics(
    "Depth (metres)", h_hyd, h_ml
)

print("\n========== VELOCITY COMPARISON ==========")
velocity_metrics = calculate_metrics(
    "Velocity (m/s)", v_hyd, v_ml
)

print("\n========== FHI COMPARISON ==========")
fhi_metrics = calculate_metrics(
    "Flood Hazard Index", fhi_hyd, fhi_ml
)


# --------------------------------------------------
# 6. Compare wet cells and hazard classes
# --------------------------------------------------

wet_hyd = h_hyd > 0.1
wet_ml = h_ml > 0.1

print("\n========== WET CELL COMPARISON ==========")
print("Hydraulic wet cells:", np.sum(wet_hyd))
print("ML wet cells       :", np.sum(wet_ml))
print(
    "Wet-cell agreement :",
    f"{np.mean(wet_hyd == wet_ml) * 100:.2f}%"
)

print("\n========== HAZARD CLASS DISTRIBUTION ==========")

for c in range(6):
    print(
        f"Class {c}: "
        f"Hydraulic = {np.sum(class_hyd == c)}, "
        f"ML = {np.sum(class_ml == c)}"
    )

class_accuracy = accuracy_score(
    class_hyd.ravel(),
    class_ml.ravel()
)

print("\nHazard class agreement:", round(class_accuracy * 100, 2), "%")

print("\nConfusion matrix:")
print(
    confusion_matrix(
        class_hyd.ravel(),
        class_ml.ravel(),
        labels=[0, 1, 2, 3, 4, 5]
    )
)


# --------------------------------------------------
# 7. Save numerical results
# --------------------------------------------------

results = pd.DataFrame([
    ["Depth", *depth_metrics],
    ["Velocity", *velocity_metrics],
    ["FHI", *fhi_metrics]
], columns=["Variable", "MAE", "RMSE", "R2"])

results.to_csv(
    os.path.join(OUTPUT_DIR, "hydraulic_ml_metrics_Q5350.csv"),
    index=False
)

# Save class comparison
class_results = pd.DataFrame({
    "Hazard_Class": range(6),
    "Hydraulic_Cells": [
        np.sum(class_hyd == c) for c in range(6)
    ],
    "ML_Cells": [
        np.sum(class_ml == c) for c in range(6)
    ]
})

class_results.to_csv(
    os.path.join(OUTPUT_DIR, "hazard_class_comparison_Q5350.csv"),
    index=False
)


# --------------------------------------------------
# 8. Generate comparison maps
# --------------------------------------------------

coord_path = os.path.join(
    ROOT, "deployment", "coordinate_grids.npz"
)

coords = np.load(coord_path)

X = coords["x"]
Y = coords["y"]

if X.shape != (h_hyd.shape[0] + 1, h_hyd.shape[1] + 1):
    raise ValueError(f"Unexpected X coordinate shape: {X.shape}")

if Y.shape != X.shape:
    raise ValueError(f"Unexpected Y coordinate shape: {Y.shape}")

fig, axes = plt.subplots(1, 3, figsize=(18, 9))

maps = [
    (class_hyd, "Hydraulic Interpolation"),
    (class_ml, "Machine Learning Prediction"),
    (
        np.where(class_hyd == class_ml, 1, 0),
        "Class Agreement (1 = Match, 0 = Mismatch)"
    )
]

for ax, (data, title) in zip(axes, maps):
    if "Agreement" in title:
        mesh = ax.pcolormesh(
            X, Y, data,
            shading="flat",
            cmap="RdYlGn",
            vmin=0, vmax=1
        )
        fig.colorbar(mesh, ax=ax, ticks=[0, 1])
    else:
        mesh = ax.pcolormesh(
            X, Y, data,
            shading="flat",
            cmap="RdYlGn_r",
            vmin=0, vmax=5
        )
        fig.colorbar(
            mesh, ax=ax,
            ticks=range(6),
            label="Hazard Class"
        )

    ax.set_title(title)
    ax.set_xlabel("Projected X coordinate")
    ax.set_ylabel("Projected Y coordinate")
    ax.set_aspect("equal")

plt.suptitle(
    f"Hydraulic vs ML Flood Hazard Comparison - Q={Q} cumecs",
    fontsize=14
)

plt.tight_layout()

map_path = os.path.join(
    OUTPUT_DIR, f"hydraulic_ml_comparison_Q{Q}.png"
)

plt.savefig(map_path, dpi=300, bbox_inches="tight")
plt.close()

print("\n========== COMPARISON COMPLETED ==========")
print("Metrics saved.")
print("Class distribution saved.")
print("Comparison map:", map_path)
print(
    "Overall hazard class agreement:",
    f"{class_accuracy * 100:.2f}%"
)
