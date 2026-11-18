
import os
import glob
import re
import numpy as np
import pandas as pd

from joblib import load
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "ML")
MODEL_DIR = os.path.join(DATA, "models")
OUTPUT_DIR = os.path.join(DATA, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load original and log models
depth_original = load(
    os.path.join(MODEL_DIR, "depth_rf.joblib")
)

velocity_original = load(
    os.path.join(MODEL_DIR, "velocity_rf.joblib")
)

depth_log = load(
    os.path.join(MODEL_DIR, "depth_rf_log.joblib")
)

velocity_log = load(
    os.path.join(MODEL_DIR, "velocity_rf_log.joblib")
)

rng = np.random.default_rng(42)

records = []

for folder in sorted(
    glob.glob(os.path.join(DATA, "BEKI_FinalQ_*_GVF*"))
):

    match = re.search(
        r"FinalQ_(\d+)", os.path.basename(folder)
    )

    if not match:
        continue

    q = int(match.group(1))

    # Same held-out scenario rule
    if (q // 200) % 5 != 1:
        continue

    h_files = glob.glob(
        os.path.join(folder, "h_avg*.xlsx")
    )

    v_files = glob.glob(
        os.path.join(folder, "rv_avg*.xlsx")
    )

    if not h_files or not v_files:
        continue

    h = pd.read_excel(
        h_files[0], header=None
    ).to_numpy(dtype=float)

    v = pd.read_excel(
        v_files[0], header=None
    ).to_numpy(dtype=float)

    if h.shape != (286, 600) or v.shape != h.shape:
        continue

    # Sample wet cells for regression comparison
    valid = np.flatnonzero((h > 0.1).ravel())

    selected = rng.choice(
        valid,
        size=min(5000, len(valid)),
        replace=False
    )

    rows, cols = np.indices(h.shape)

    features = np.column_stack([
        np.full(len(selected), q),
        rows.ravel()[selected],
        cols.ravel()[selected]
    ])

    actual_depth = h.ravel()[selected]
    actual_velocity = v.ravel()[selected]

    # Original predictions
    pred_depth_original = np.maximum(
        depth_original.predict(features), 0
    )

    pred_velocity_original = np.maximum(
        velocity_original.predict(features), 0
    )

    # Log model predictions
    pred_depth_log = np.maximum(
        np.expm1(depth_log.predict(features)), 0
    )

    pred_velocity_log = np.maximum(
        np.expm1(velocity_log.predict(features)), 0
    )

    models = [
        ("Original", pred_depth_original, pred_velocity_original),
        ("Log", pred_depth_log, pred_velocity_log)
    ]

    for model_name, pdp, pvp in models:

        for variable, actual, predicted in [
            ("Depth", actual_depth, pdp),
            ("Velocity", actual_velocity, pvp)
        ]:

            records.append({
                "Discharge": q,
                "Model": model_name,
                "Variable": variable,
                "MAE": mean_absolute_error(
                    actual, predicted
                ),
                "RMSE": np.sqrt(
                    mean_squared_error(actual, predicted)
                ),
                "R2": r2_score(actual, predicted),
                "Actual_95th": np.percentile(actual, 95),
                "Predicted_95th": np.percentile(predicted, 95),
                "Actual_99th": np.percentile(actual, 99),
                "Predicted_99th": np.percentile(predicted, 99),
                "Actual_Max": actual.max(),
                "Predicted_Max": predicted.max()
            })

    print("Evaluated Q =", q)


df = pd.DataFrame(records)

output_path = os.path.join(
    OUTPUT_DIR,
    "original_vs_log_regression_comparison.csv"
)

df.to_csv(output_path, index=False)

print("\n========== ORIGINAL VS LOG MODEL ==========")

print(
    df.groupby(["Model", "Variable"])[
        ["MAE", "RMSE", "R2"]
    ].mean().round(4)
)

print("\n========== HIGH-DISCHARGE COMPARISON ==========")

print(
    df[df["Discharge"] >= 6200][
        [
            "Discharge", "Model", "Variable",
            "MAE", "RMSE", "R2",
            "Actual_99th", "Predicted_99th",
            "Actual_Max", "Predicted_Max"
        ]
    ].to_string(index=False)
)

print("\nSaved:", output_path)
