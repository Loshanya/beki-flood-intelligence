
import os
import glob
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix
)

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "ML")
MODEL_DIR = os.path.join(DATA_DIR, "models")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DISCHARGES = [1200, 2200, 3200, 4200, 6200, 7200, 8200, 9200]

ROWS = 286
COLS = 600

# ==========================================
# LOAD MODELS
# ==========================================

print("Loading V3 models...")

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

print("All V3 models loaded successfully.")

# ==========================================
# FIND HYDRAULIC SCENARIO
# ==========================================

def find_scenario_folder(q):

    pattern = os.path.join(
        DATA_DIR, f"BEKI_FinalQ_{q}_GVF*"
    )

    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(
            f"Scenario folder missing for Q={q}"
        )

    return matches[0]


def find_excel_file(folder, prefix):

    matches = [
        f for f in os.listdir(folder)
        if f.lower().startswith(prefix.lower())
        and f.lower().endswith((".xlsx", ".xls"))
    ]

    if not matches:
        raise FileNotFoundError(
            f"{prefix} file missing in {folder}"
        )

    exact = prefix.lower() + ".xlsx"

    for f in matches:
        if f.lower() == exact:
            return os.path.join(folder, f)

    return os.path.join(folder, matches[0])


def load_hydraulic_data(q):

    folder = find_scenario_folder(q)

    h_file = find_excel_file(folder, "h_avg")
    v_file = find_excel_file(folder, "rv_avg")

    depth = pd.read_excel(
        h_file, header=None
    ).to_numpy(dtype=float)

    velocity = pd.read_excel(
        v_file, header=None
    ).to_numpy(dtype=float)

    if depth.shape != (ROWS, COLS):
        raise ValueError(
            f"Unexpected depth shape for Q={q}: {depth.shape}"
        )

    if velocity.shape != (ROWS, COLS):
        raise ValueError(
            f"Unexpected velocity shape for Q={q}: {velocity.shape}"
        )

    return depth, velocity


# ==========================================
# HAZARD CLASSIFICATION
# ==========================================

def calculate_hazard(depth, velocity):

    fhi = depth * (velocity + 0.5)

    hazard = np.zeros(depth.shape, dtype=np.int8)

    wet = depth > 0.1

    hazard[wet & (fhi <= 0.5)] = 1
    hazard[wet & (fhi > 0.5) & (fhi <= 1)] = 2
    hazard[wet & (fhi > 1) & (fhi <= 2)] = 3
    hazard[wet & (fhi > 2) & (fhi <= 3)] = 4
    hazard[wet & (fhi > 3)] = 5

    return fhi, hazard


# ==========================================
# PREDICT V3 FOR ONE DISCHARGE
# ==========================================

def predict_v3(q):

    row_grid, col_grid = np.indices((ROWS, COLS))

    features = np.column_stack([
        np.full(ROWS * COLS, q),
        row_grid.ravel(),
        col_grid.ravel()
    ])

    # Wet/dry prediction
    probabilities = wet_model.predict_proba(features)[:, 1]

    wet_mask = probabilities >= wet_threshold

    # Log-transformed regression predictions
    depth_log = depth_model.predict(features)
    velocity_log = velocity_model.predict(features)

    depth_all = np.maximum(np.expm1(depth_log), 0)
    velocity_all = np.maximum(np.expm1(velocity_log), 0)

    # Apply wet/dry mask
    depth = np.zeros(ROWS * COLS)
    velocity = np.zeros(ROWS * COLS)

    depth[wet_mask] = np.maximum(depth_all[wet_mask], 0.1)
    velocity[wet_mask] = velocity_all[wet_mask]

    depth = depth.reshape(ROWS, COLS)
    velocity = velocity.reshape(ROWS, COLS)

    fhi, hazard = calculate_hazard(depth, velocity)

    return depth, velocity, fhi, hazard


# ==========================================
# METRICS
# ==========================================

def regression_metrics(actual, predicted):

    return (
        mean_absolute_error(actual, predicted),
        np.sqrt(mean_squared_error(actual, predicted)),
        r2_score(actual, predicted)
    )


# ==========================================
# MULTI-DISCHARGE EVALUATION
# ==========================================

results = []

combined_confusion = np.zeros((6, 6), dtype=np.int64)

for q in DISCHARGES:

    print("\n" + "=" * 55)
    print(f"Evaluating V3 at Q = {q}")
    print("=" * 55)

    # Load actual hydraulic values
    actual_depth, actual_velocity = load_hydraulic_data(q)

    # Generate V3 predictions
    pred_depth, pred_velocity, pred_fhi, pred_hazard = predict_v3(q)

    # Actual FHI and hazard
    actual_fhi, actual_hazard = calculate_hazard(
        actual_depth, actual_velocity
    )

    actual_wet = actual_depth > 0.1
    predicted_wet = pred_depth > 0.1

    # Depth metrics: all cells
    d_mae, d_rmse, d_r2 = regression_metrics(
        actual_depth.ravel(),
        pred_depth.ravel()
    )

    # Depth metrics: actual wet cells only
    dw_mae, dw_rmse, dw_r2 = regression_metrics(
        actual_depth[actual_wet],
        pred_depth[actual_wet]
    )

    # Velocity metrics: all cells
    v_mae, v_rmse, v_r2 = regression_metrics(
        actual_velocity.ravel(),
        pred_velocity.ravel()
    )

    # Velocity metrics: actual wet cells only
    vw_mae, vw_rmse, vw_r2 = regression_metrics(
        actual_velocity[actual_wet],
        pred_velocity[actual_wet]
    )

    # FHI metrics
    fhi_mae, fhi_rmse, fhi_r2 = regression_metrics(
        actual_fhi.ravel(),
        pred_fhi.ravel()
    )

    # Classification metrics
    wet_accuracy = accuracy_score(
        actual_wet.ravel(),
        predicted_wet.ravel()
    )

    wet_f1 = f1_score(
        actual_wet.ravel(),
        predicted_wet.ravel(),
        zero_division=0
    )

    hazard_accuracy = accuracy_score(
        actual_hazard.ravel(),
        pred_hazard.ravel()
    )

    hazard_macro_f1 = f1_score(
        actual_hazard.ravel(),
        pred_hazard.ravel(),
        labels=[0, 1, 2, 3, 4, 5],
        average="macro",
        zero_division=0
    )

    # Combine confusion matrices
    cm = confusion_matrix(
        actual_hazard.ravel(),
        pred_hazard.ravel(),
        labels=[0, 1, 2, 3, 4, 5]
    )

    combined_confusion += cm

    # Record metrics
    row = {
        "Discharge": q,

        "Depth_MAE": d_mae,
        "Depth_RMSE": d_rmse,
        "Depth_R2": d_r2,

        "Wet_Depth_MAE": dw_mae,
        "Wet_Depth_RMSE": dw_rmse,
        "Wet_Depth_R2": dw_r2,

        "Velocity_MAE": v_mae,
        "Velocity_RMSE": v_rmse,
        "Velocity_R2": v_r2,

        "Wet_Velocity_MAE": vw_mae,
        "Wet_Velocity_RMSE": vw_rmse,
        "Wet_Velocity_R2": vw_r2,

        "FHI_MAE": fhi_mae,
        "FHI_RMSE": fhi_rmse,
        "FHI_R2": fhi_r2,

        "Wet_Accuracy": wet_accuracy,
        "Wet_F1": wet_f1,

        "Hazard_Accuracy": hazard_accuracy,
        "Hazard_Macro_F1": hazard_macro_f1,

        "Actual_Wet_Cells": int(np.sum(actual_wet)),
        "Predicted_Wet_Cells": int(np.sum(predicted_wet))
    }

    results.append(row)

    print(f"Depth MAE:       {d_mae:.4f}")
    print(f"Depth R2:        {d_r2:.4f}")
    print(f"Wet depth MAE:   {dw_mae:.4f}")

    print(f"Velocity MAE:    {v_mae:.4f}")
    print(f"Velocity R2:     {v_r2:.4f}")
    print(f"Wet velocity MAE:{vw_mae:.4f}")

    print(f"FHI MAE:         {fhi_mae:.4f}")
    print(f"FHI R2:          {fhi_r2:.4f}")

    print(f"Wet accuracy:    {wet_accuracy:.2%}")
    print(f"Wet F1:          {wet_f1:.4f}")

    print(f"Hazard accuracy: {hazard_accuracy:.2%}")
    print(f"Hazard Macro F1: {hazard_macro_f1:.4f}")

    print("Actual wet cells:", np.sum(actual_wet))
    print("Predicted wet cells:", np.sum(predicted_wet))


# ==========================================
# SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

csv_file = os.path.join(
    OUTPUT_DIR,
    "multi_discharge_validation_v3.csv"
)

results_df.to_csv(csv_file, index=False)

confusion_file = os.path.join(
    OUTPUT_DIR,
    "multi_discharge_confusion_matrix_v3.csv"
)

pd.DataFrame(
    combined_confusion,
    index=[f"Actual_{i}" for i in range(6)],
    columns=[f"Predicted_{i}" for i in range(6)]
).to_csv(confusion_file)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 70)
print("FINAL V3 MULTI-DISCHARGE VALIDATION")
print("=" * 70)

columns = [
    "Discharge",
    "Depth_MAE",
    "Depth_R2",
    "Velocity_MAE",
    "Velocity_R2",
    "FHI_MAE",
    "FHI_R2",
    "Wet_Accuracy",
    "Hazard_Accuracy",
    "Hazard_Macro_F1"
]

print(
    results_df[columns].round(4).to_string(index=False)
)

print("\nSaved results:")
print(csv_file)
print(confusion_file)

print("\nV3 multi-discharge evaluation completed.")
