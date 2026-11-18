
import os
import glob
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    f1_score
)

# ==========================================
# PATHS AND SETTINGS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "ML")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")

Q = 5350
Q_LOW = 5200
Q_HIGH = 5400

V2_FILE = os.path.join(
    OUTPUT_DIR, f"ml_prediction_v2_Q{Q}.npz"
)

V3_FILE = os.path.join(
    OUTPUT_DIR, f"ml_prediction_v3_Q{Q}.npz"
)

# ==========================================
# READ HYDRAULIC SCENARIO FILES
# ==========================================

def find_scenario_folder(q):
    pattern = os.path.join(
        DATA_DIR, f"BEKI_FinalQ_{q}_GVF*"
    )

    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(
            f"Scenario folder not found for Q={q}: {pattern}"
        )

    return matches[0]


def find_excel_file(folder, prefix):
    files = os.listdir(folder)

    matches = [
        f for f in files
        if f.lower().startswith(prefix.lower())
        and f.lower().endswith((".xlsx", ".xls"))
    ]

    if not matches:
        raise FileNotFoundError(
            f"{prefix} Excel file not found in {folder}"
        )

    # Prefer exact standard filename if available
    exact = prefix.lower() + ".xlsx"

    for f in matches:
        if f.lower() == exact:
            return os.path.join(folder, f)

    return os.path.join(folder, matches[0])


def load_hydraulic_scenario(q):
    folder = find_scenario_folder(q)

    h_file = find_excel_file(folder, "h_avg")
    v_file = find_excel_file(folder, "rv_avg")

    depth = pd.read_excel(
        h_file, header=None
    ).to_numpy(dtype=float)

    velocity = pd.read_excel(
        v_file, header=None
    ).to_numpy(dtype=float)

    print(f"Loaded hydraulic scenario Q={q}")
    print("Depth shape:", depth.shape)
    print("Velocity shape:", velocity.shape)

    return depth, velocity


# ==========================================
# HYDRAULIC INTERPOLATION
# ==========================================

print("\nLoading hydraulic reference...")

h_low, v_low = load_hydraulic_scenario(Q_LOW)
h_high, v_high = load_hydraulic_scenario(Q_HIGH)

if h_low.shape != h_high.shape:
    raise ValueError("Hydraulic depth grid shapes do not match.")

if v_low.shape != v_high.shape:
    raise ValueError("Hydraulic velocity grid shapes do not match.")

if h_low.shape != v_low.shape:
    raise ValueError("Depth and velocity grid shapes do not match.")

weight = (Q - Q_LOW) / (Q_HIGH - Q_LOW)

hydraulic_depth = (
    h_low * (1 - weight) + h_high * weight
)

hydraulic_velocity = (
    v_low * (1 - weight) + v_high * weight
)

# ==========================================
# HYDRAULIC FHI AND HAZARD CLASSES
# ==========================================

hydraulic_fhi = hydraulic_depth * (
    hydraulic_velocity + 0.5
)

hydraulic_hazard = np.zeros(
    hydraulic_depth.shape, dtype=np.int8
)

hydraulic_wet = hydraulic_depth > 0.1

hydraulic_hazard[
    hydraulic_wet & (hydraulic_fhi <= 0.5)
] = 1

hydraulic_hazard[
    hydraulic_wet & (hydraulic_fhi > 0.5)
    & (hydraulic_fhi <= 1)
] = 2

hydraulic_hazard[
    hydraulic_wet & (hydraulic_fhi > 1)
    & (hydraulic_fhi <= 2)
] = 3

hydraulic_hazard[
    hydraulic_wet & (hydraulic_fhi > 2)
    & (hydraulic_fhi <= 3)
] = 4

hydraulic_hazard[
    hydraulic_wet & (hydraulic_fhi > 3)
] = 5


# ==========================================
# LOAD V2 AND V3
# ==========================================

print("\nLoading V2 and V3 predictions...")

v2_data = np.load(V2_FILE)
v3_data = np.load(V3_FILE)



def get_prediction(data):

    depth = data["depth"]
    velocity = data["velocity"]

    # Support different key names across saved versions
    if "hazard" in data.files:
        hazard = data["hazard"]
    elif "classes" in data.files:
        hazard = data["classes"]
    elif "hazard_classes" in data.files:
        hazard = data["hazard_classes"]
    else:
        # Recalculate from saved depth and velocity
        fhi = depth * (velocity + 0.5)

        hazard = np.zeros(depth.shape, dtype=np.int8)
        wet = depth > 0.1

        hazard[wet & (fhi <= 0.5)] = 1
        hazard[wet & (fhi > 0.5) & (fhi <= 1)] = 2
        hazard[wet & (fhi > 1) & (fhi <= 2)] = 3
        hazard[wet & (fhi > 2) & (fhi <= 3)] = 4
        hazard[wet & (fhi > 3)] = 5

    return depth, velocity, hazard



v2_depth, v2_velocity, v2_hazard = get_prediction(v2_data)
v3_depth, v3_velocity, v3_hazard = get_prediction(v3_data)

for name, array in [
    ("V2 depth", v2_depth),
    ("V2 velocity", v2_velocity),
    ("V2 hazard", v2_hazard),
    ("V3 depth", v3_depth),
    ("V3 velocity", v3_velocity),
    ("V3 hazard", v3_hazard)
]:
    if array.shape != hydraulic_depth.shape:
        raise ValueError(
            f"{name} shape {array.shape} does not match "
            f"hydraulic shape {hydraulic_depth.shape}"
        )


# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, depth, velocity, hazard):

    print(f"\n========== {name} ==========")

    depth_mae = mean_absolute_error(
        hydraulic_depth.ravel(), depth.ravel()
    )

    depth_rmse = np.sqrt(mean_squared_error(
        hydraulic_depth.ravel(), depth.ravel()
    ))

    depth_r2 = r2_score(
        hydraulic_depth.ravel(), depth.ravel()
    )

    velocity_mae = mean_absolute_error(
        hydraulic_velocity.ravel(), velocity.ravel()
    )

    velocity_rmse = np.sqrt(mean_squared_error(
        hydraulic_velocity.ravel(), velocity.ravel()
    ))

    velocity_r2 = r2_score(
        hydraulic_velocity.ravel(), velocity.ravel()
    )

    fhi = depth * (velocity + 0.5)

    fhi_mae = mean_absolute_error(
        hydraulic_fhi.ravel(), fhi.ravel()
    )

    wet_accuracy = accuracy_score(
        hydraulic_wet.ravel(),
        (depth > 0.1).ravel()
    )

    hazard_accuracy = accuracy_score(
        hydraulic_hazard.ravel(),
        hazard.ravel()
    )

    macro_f1 = f1_score(
        hydraulic_hazard.ravel(),
        hazard.ravel(),
        average="macro",
        zero_division=0
    )

    print(f"Depth MAE:       {depth_mae:.4f}")
    print(f"Depth RMSE:      {depth_rmse:.4f}")
    print(f"Depth R2:        {depth_r2:.4f}")

    print(f"Velocity MAE:    {velocity_mae:.4f}")
    print(f"Velocity RMSE:   {velocity_rmse:.4f}")
    print(f"Velocity R2:     {velocity_r2:.4f}")

    print(f"FHI MAE:         {fhi_mae:.4f}")
    print(f"Wet accuracy:    {wet_accuracy:.4%}")
    print(f"Hazard accuracy: {hazard_accuracy:.4%}")
    print(f"Hazard Macro F1: {macro_f1:.4f}")

    return {
        "Model": name,
        "Depth_MAE": depth_mae,
        "Depth_RMSE": depth_rmse,
        "Depth_R2": depth_r2,
        "Velocity_MAE": velocity_mae,
        "Velocity_RMSE": velocity_rmse,
        "Velocity_R2": velocity_r2,
        "FHI_MAE": fhi_mae,
        "Wet_Accuracy": wet_accuracy,
        "Hazard_Accuracy": hazard_accuracy,
        "Hazard_Macro_F1": macro_f1
    }


# ==========================================
# COMPARE BOTH MODELS
# ==========================================

results = []

results.append(
    evaluate_model(
        "V2",
        v2_depth,
        v2_velocity,
        v2_hazard
    )
)

results.append(
    evaluate_model(
        "V3",
        v3_depth,
        v3_velocity,
        v3_hazard
    )
)

# ==========================================
# CONFUSION MATRICES
# ==========================================

cm_v2 = confusion_matrix(
    hydraulic_hazard.ravel(),
    v2_hazard.ravel(),
    labels=[0, 1, 2, 3, 4, 5]
)

cm_v3 = confusion_matrix(
    hydraulic_hazard.ravel(),
    v3_hazard.ravel(),
    labels=[0, 1, 2, 3, 4, 5]
)

print("\nV2 CONFUSION MATRIX")
print(cm_v2)

print("\nV3 CONFUSION MATRIX")
print(cm_v3)

# ==========================================
# SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

output_csv = os.path.join(
    OUTPUT_DIR,
    "v2_vs_v3_comparison.csv"
)

results_df.to_csv(output_csv, index=False)

pd.DataFrame(
    cm_v2,
    index=[f"Actual_{i}" for i in range(6)],
    columns=[f"Predicted_{i}" for i in range(6)]
).to_csv(
    os.path.join(OUTPUT_DIR, "v2_confusion_matrix.csv")
)

pd.DataFrame(
    cm_v3,
    index=[f"Actual_{i}" for i in range(6)],
    columns=[f"Predicted_{i}" for i in range(6)]
).to_csv(
    os.path.join(OUTPUT_DIR, "v3_confusion_matrix.csv")
)

print("\n========== FINAL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nResults saved:", output_csv)
print("\nComparison completed.")
