
import os
import glob
import re
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "ML")
MODEL_DIR = os.path.join(DATA, "models")
OUTPUT_DIR = os.path.join(DATA, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)

X_train = []
X_test = []

depth_train = []
depth_test = []

velocity_train = []
velocity_test = []

print("Loading hydraulic scenarios...")

folders = sorted(
    glob.glob(os.path.join(DATA, "BEKI_FinalQ_*_GVF*"))
)

for folder in folders:

    name = os.path.basename(folder)
    match = re.search(r"FinalQ_(\d+)", name)

    if not match:
        continue

    q = int(match.group(1))

    h_files = glob.glob(os.path.join(folder, "h_avg*.xlsx"))
    v_files = glob.glob(os.path.join(folder, "rv_avg*.xlsx"))

    if not h_files or not v_files:
        continue

    try:
        h = pd.read_excel(
            h_files[0], header=None
        ).to_numpy(dtype=float)

        v = pd.read_excel(
            v_files[0], header=None
        ).to_numpy(dtype=float)

        if h.shape != (286, 600) or v.shape != h.shape:
            print("Skipping incompatible grid:", q)
            continue

        # Sample wet cells only for hydraulic regression
        wet = h > 0.1
        valid_indices = np.flatnonzero(wet.ravel())

        n = min(5000, len(valid_indices))

        if n == 0:
            continue

        selected = rng.choice(
            valid_indices,
            size=n,
            replace=False
        )

        rows, cols = np.indices(h.shape)

        r = rows.ravel()[selected]
        c = cols.ravel()[selected]

        features = np.column_stack([
            np.full(n, q),
            r,
            c
        ])

        depth_values = h.ravel()[selected]
        velocity_values = v.ravel()[selected]

        # Same scenario holdout logic as previous regressors
        scenario_number = q // 200

        if scenario_number % 5 == 1:

            X_test.append(features)
            depth_test.append(depth_values)
            velocity_test.append(velocity_values)

        else:

            X_train.append(features)
            depth_train.append(depth_values)
            velocity_train.append(velocity_values)

        print(
            f"Q={q}: sampled {n} wet cells"
        )

    except Exception as e:
        print("Error processing", q, e)


X_train = np.vstack(X_train)
X_test = np.vstack(X_test)

depth_train = np.concatenate(depth_train)
depth_test = np.concatenate(depth_test)

velocity_train = np.concatenate(velocity_train)
velocity_test = np.concatenate(velocity_test)

print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))


# ==================================================
# DEPTH MODEL
# ==================================================

print("\nTraining log-transformed depth model...")

depth_model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=2,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)

depth_model.fit(
    X_train,
    np.log1p(depth_train)
)

depth_predictions = np.expm1(
    depth_model.predict(X_test)
)

depth_predictions = np.maximum(
    depth_predictions, 0
)

print("\nDEPTH RESULTS")

print(
    "MAE:",
    mean_absolute_error(
        depth_test, depth_predictions
    )
)

print(
    "RMSE:",
    np.sqrt(
        mean_squared_error(
            depth_test, depth_predictions
        )
    )
)

print(
    "R2:",
    r2_score(
        depth_test, depth_predictions
    )
)


# ==================================================
# VELOCITY MODEL
# ==================================================

print("\nTraining log-transformed velocity model...")

velocity_model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=2,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)

velocity_model.fit(
    X_train,
    np.log1p(velocity_train)
)

velocity_predictions = np.expm1(
    velocity_model.predict(X_test)
)

velocity_predictions = np.maximum(
    velocity_predictions, 0
)

print("\nVELOCITY RESULTS")

print(
    "MAE:",
    mean_absolute_error(
        velocity_test, velocity_predictions
    )
)

print(
    "RMSE:",
    np.sqrt(
        mean_squared_error(
            velocity_test, velocity_predictions
        )
    )
)

print(
    "R2:",
    r2_score(
        velocity_test, velocity_predictions
    )
)


# ==================================================
# SAVE MODELS
# ==================================================

dump(
    depth_model,
    os.path.join(
        MODEL_DIR,
        "depth_rf_log.joblib"
    )
)

dump(
    velocity_model,
    os.path.join(
        MODEL_DIR,
        "velocity_rf_log.joblib"
    )
)

print("\nExperimental models saved successfully.")

print(
    os.path.join(
        MODEL_DIR,
        "depth_rf_log.joblib"
    )
)

print(
    os.path.join(
        MODEL_DIR,
        "velocity_rf_log.joblib"
    )
)
