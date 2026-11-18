
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix
)

# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "ML")
MODEL_DIR = os.path.join(DATA_DIR, "models")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scenarios held out by the existing training split
TEST_DISCHARGES = [
    1200, 2200, 3200, 4200,
    6200, 7200, 8200, 9200
]

ROWS = 286
COLS = 600

print("Loading trained ML models...")

wet_bundle = load(
    os.path.join(MODEL_DIR, "wet_classifier.joblib")
)

wet_model = wet_bundle["model"]
wet_threshold = wet_bundle["threshold"]

depth_model = load(
    os.path.join(MODEL_DIR, "depth_rf.joblib")
)

velocity_model = load(
    os.path.join(MODEL_DIR, "velocity_rf.joblib")
)

print("All models loaded successfully.")
print("Wet classifier threshold:", wet_threshold)


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def find_folder(q):

    matches = glob.glob(
        os.path.join(
            DATA_DIR,
            f"BEKI_FinalQ_{q}_GVF*"
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"Hydraulic folder missing for Q={q}"
        )

    return matches[0]


def read_grid(folder, prefix):

    matches = glob.glob(
        os.path.join(folder, prefix + "*.xlsx")
    )

    if not matches:
        raise FileNotFoundError(
            f"{prefix} file missing in {folder}"
        )

    return pd.read_excel(
        matches[0],
        header=None
    ).to_numpy(dtype=float)


def calculate_classes(depth, velocity, wet_mask):

    fhi = depth * (velocity + 0.5)

    classes = np.zeros(
        depth.shape,
        dtype=np.uint8
    )

    classes[wet_mask & (fhi <= 0.5)] = 1
    classes[wet_mask & (fhi > 0.5) & (fhi <= 1)] = 2
    classes[wet_mask & (fhi > 1) & (fhi <= 2)] = 3
    classes[wet_mask & (fhi > 2) & (fhi <= 3)] = 4
    classes[wet_mask & (fhi > 3)] = 5

    return fhi, classes


# Prepare spatial features once
row_grid, col_grid = np.indices((ROWS, COLS))

spatial_features = np.column_stack([
    row_grid.ravel(),
    col_grid.ravel()
])


# ==================================================
# 3. EVALUATE EACH HELD-OUT DISCHARGE
# ==================================================

results = []

aggregate_confusion = np.zeros(
    (6, 6),
    dtype=np.int64
)

for Q in TEST_DISCHARGES:

    print("\n" + "=" * 55)
    print(f"EVALUATING DISCHARGE: {Q} CUMecs")
    print("=" * 55)

    folder = find_folder(Q)

    # Load exact hydraulic reference
    h_hyd = read_grid(folder, "h_avg")
    v_hyd = read_grid(folder, "rv_avg")

    if h_hyd.shape != (ROWS, COLS):
        raise ValueError(
            f"Unexpected depth grid shape at Q={Q}: "
            f"{h_hyd.shape}"
        )

    if v_hyd.shape != h_hyd.shape:
        raise ValueError(
            f"Velocity grid mismatch at Q={Q}"
        )

    # Create features for this discharge
    q_column = np.full(
        ROWS * COLS,
        Q
    )

    features = np.column_stack([
        q_column,
        spatial_features
    ])

    # ----------------------------------------------
    # ML wet/dry classification
    # ----------------------------------------------

    wet_prob = wet_model.predict_proba(
        features
    )[:, 1]

    wet_ml = (
        wet_prob >= wet_threshold
    ).reshape(ROWS, COLS)

    # ----------------------------------------------
    # ML depth and velocity regression
    # ----------------------------------------------

    h_ml = depth_model.predict(
        features
    ).reshape(ROWS, COLS)

    v_ml = velocity_model.predict(
        features
    ).reshape(ROWS, COLS)

    h_ml = np.maximum(h_ml, 0)
    v_ml = np.maximum(v_ml, 0)

    # Apply wet/dry classification
    h_ml[~wet_ml] = 0.1
    v_ml[~wet_ml] = 0.0

    # ----------------------------------------------
    # Calculate FHI and hazard classes
    # ----------------------------------------------

    wet_hyd = h_hyd > 0.1

    fhi_hyd, class_hyd = calculate_classes(
        h_hyd,
        v_hyd,
        wet_hyd
    )

    fhi_ml, class_ml = calculate_classes(
        h_ml,
        v_ml,
        wet_ml
    )

    # ----------------------------------------------
    # Calculate numerical metrics
    # ----------------------------------------------

    depth_mae = mean_absolute_error(
        h_hyd.ravel(),
        h_ml.ravel()
    )

    depth_rmse = np.sqrt(
        mean_squared_error(
            h_hyd.ravel(),
            h_ml.ravel()
        )
    )

    depth_r2 = r2_score(
        h_hyd.ravel(),
        h_ml.ravel()
    )

    velocity_mae = mean_absolute_error(
        v_hyd.ravel(),
        v_ml.ravel()
    )

    velocity_rmse = np.sqrt(
        mean_squared_error(
            v_hyd.ravel(),
            v_ml.ravel()
        )
    )

    velocity_r2 = r2_score(
        v_hyd.ravel(),
        v_ml.ravel()
    )

    fhi_mae = mean_absolute_error(
        fhi_hyd.ravel(),
        fhi_ml.ravel()
    )

    fhi_rmse = np.sqrt(
        mean_squared_error(
            fhi_hyd.ravel(),
            fhi_ml.ravel()
        )
    )

    fhi_r2 = r2_score(
        fhi_hyd.ravel(),
        fhi_ml.ravel()
    )

    # ----------------------------------------------
    # Wet/dry evaluation
    # ----------------------------------------------

    wet_accuracy = accuracy_score(
        wet_hyd.ravel(),
        wet_ml.ravel()
    )

    wet_f1 = f1_score(
        wet_hyd.ravel(),
        wet_ml.ravel(),
        zero_division=0
    )

    hydraulic_wet_cells = int(wet_hyd.sum())
    ml_wet_cells = int(wet_ml.sum())

    # ----------------------------------------------
    # Hazard class evaluation
    # ----------------------------------------------

    class_accuracy = accuracy_score(
        class_hyd.ravel(),
        class_ml.ravel()
    )

    class_macro_f1 = f1_score(
        class_hyd.ravel(),
        class_ml.ravel(),
        labels=[0, 1, 2, 3, 4, 5],
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        class_hyd.ravel(),
        class_ml.ravel(),
        labels=[0, 1, 2, 3, 4, 5]
    )

    aggregate_confusion += cm

    # ----------------------------------------------
    # Store results
    # ----------------------------------------------

    results.append({
        "Discharge": Q,

        "Hydraulic_Wet_Cells": hydraulic_wet_cells,
        "ML_Wet_Cells": ml_wet_cells,

        "Wet_Cell_Difference": (
            ml_wet_cells - hydraulic_wet_cells
        ),

        "Wet_Accuracy": wet_accuracy,
        "Wet_F1": wet_f1,

        "Depth_MAE": depth_mae,
        "Depth_RMSE": depth_rmse,
        "Depth_R2": depth_r2,

        "Velocity_MAE": velocity_mae,
        "Velocity_RMSE": velocity_rmse,
        "Velocity_R2": velocity_r2,

        "FHI_MAE": fhi_mae,
        "FHI_RMSE": fhi_rmse,
        "FHI_R2": fhi_r2,

        "Hazard_Class_Accuracy": class_accuracy,
        "Hazard_Class_Macro_F1": class_macro_f1
    })

    print(f"Hydraulic wet cells : {hydraulic_wet_cells}")
    print(f"ML wet cells        : {ml_wet_cells}")

    print(f"Depth MAE           : {depth_mae:.4f}")
    print(f"Depth R2            : {depth_r2:.4f}")

    print(f"Velocity MAE        : {velocity_mae:.4f}")
    print(f"Velocity R2         : {velocity_r2:.4f}")

    print(f"FHI MAE             : {fhi_mae:.4f}")
    print(f"FHI R2              : {fhi_r2:.4f}")

    print(f"Wet-cell accuracy   : {wet_accuracy * 100:.2f}%")
    print(f"Wet-cell F1         : {wet_f1:.4f}")

    print(
        f"Hazard agreement    : "
        f"{class_accuracy * 100:.2f}%"
    )

    print(f"Hazard macro F1     : {class_macro_f1:.4f}")


# ==================================================
# 4. SAVE CONSOLIDATED RESULTS
# ==================================================

df = pd.DataFrame(results)

csv_path = os.path.join(
    OUTPUT_DIR,
    "multi_discharge_validation_v2.csv"
)

df.to_csv(csv_path, index=False)

print("\n" + "=" * 55)
print("MULTI-DISCHARGE VALIDATION SUMMARY")
print("=" * 55)

print(
    df[
        [
            "Discharge",
            "Depth_MAE",
            "Velocity_MAE",
            "FHI_MAE",
            "Wet_Accuracy",
            "Hazard_Class_Accuracy",
            "Hazard_Class_Macro_F1"
        ]
    ].to_string(index=False)
)

# Save combined hazard confusion matrix
cm_df = pd.DataFrame(
    aggregate_confusion,
    index=[f"Actual_{i}" for i in range(6)],
    columns=[f"Predicted_{i}" for i in range(6)]
)

cm_path = os.path.join(
    OUTPUT_DIR,
    "multi_discharge_confusion_matrix_v2.csv"
)

cm_df.to_csv(cm_path)


# ==================================================
# 5. GENERATE PERFORMANCE GRAPHS
# ==================================================

# Graph 1: Wet-cell and hazard-class accuracy
plt.figure(figsize=(10, 6))

plt.plot(
    df["Discharge"],
    df["Wet_Accuracy"] * 100,
    marker="o",
    label="Wet/Dry Accuracy"
)

plt.plot(
    df["Discharge"],
    df["Hazard_Class_Accuracy"] * 100,
    marker="s",
    label="Hazard Class Agreement"
)

plt.xlabel("Discharge (cumecs)")
plt.ylabel("Accuracy (%)")
plt.title("V2 Model Performance Across Discharges")
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "multi_discharge_accuracy_v2.png"
    ),
    dpi=300
)

plt.close()


# Graph 2: Depth and velocity errors
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(
    df["Discharge"],
    df["Depth_MAE"],
    marker="o",
    label="Depth MAE"
)

ax1.set_xlabel("Discharge (cumecs)")
ax1.set_ylabel("Depth MAE (m)")
ax1.grid(True)

ax2 = ax1.twinx()

ax2.plot(
    df["Discharge"],
    df["Velocity_MAE"],
    marker="s",
    label="Velocity MAE"
)

ax2.set_ylabel("Velocity MAE (m/s)")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="best"
)

plt.title("Depth and Velocity Prediction Errors")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "multi_discharge_errors_v2.png"
    ),
    dpi=300
)

plt.close()


# ==================================================
# 6. FINAL OUTPUT
# ==================================================

print("\n" + "=" * 55)
print("MULTI-DISCHARGE VALIDATION COMPLETED")
print("=" * 55)

print("\nGenerated files:")

print(csv_path)
print(cm_path)

print(
    os.path.join(
        OUTPUT_DIR,
        "multi_discharge_accuracy_v2.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "multi_discharge_errors_v2.png"
    )
)
