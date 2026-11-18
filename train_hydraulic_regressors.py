
from pathlib import Path
import re
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent / "ML"
OUTPUT = ROOT / "outputs"
MODELS = ROOT / "models"

OUTPUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

SAMPLES = 3000
rng = np.random.default_rng(42)


def get_q(name):
    m = re.search(r"BEKI_FinalQ_(\d+)_", name)
    return int(m.group(1)) if m else None


def find_file(folder, prefix):
    files = [
        p for p in folder.glob("*.xlsx")
        if p.name.lower().startswith(prefix)
        and "_org" not in p.name.lower()
    ]
    return files[0] if files else None


def read_grid(path):
    return pd.read_excel(path, header=None).apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)


X_train, H_train, V_train = [], [], []
X_test, H_test, V_test = [], [], []

folders = sorted(
    [p for p in ROOT.iterdir()
     if p.is_dir() and get_q(p.name) is not None],
    key=lambda p: get_q(p.name)
)

print("Preparing hydraulic dataset...")

for folder in folders:
    q = get_q(folder.name)

    hf = find_file(folder, "h_avg")
    vf = find_file(folder, "rv_avg")

    if hf is None or vf is None:
        print("Skipping Q =", q)
        continue

    h = read_grid(hf)
    v = read_grid(vf)

    if h.shape != (286, 600) or v.shape != h.shape:
        print("Invalid grid:", q)
        continue

    valid = np.isfinite(h) & np.isfinite(v)
    indices = np.flatnonzero(valid)

    if len(indices) > SAMPLES:
        indices = rng.choice(indices, SAMPLES, replace=False)

    rows, cols = np.unravel_index(indices, h.shape)

    X = np.column_stack([
        np.full(len(indices), q),
        rows,
        cols
    ])

    depths = h.ravel()[indices]
    velocities = v.ravel()[indices]

    # Hold out every fifth scenario for evaluation
    scenario_number = q // 200

    if scenario_number % 5 == 1:
        X_test.append(X)
        H_test.append(depths)
        V_test.append(velocities)
    else:
        X_train.append(X)
        H_train.append(depths)
        V_train.append(velocities)

    print("Processed Q =", q)

X_train = np.vstack(X_train)
X_test = np.vstack(X_test)

H_train = np.concatenate(H_train)
H_test = np.concatenate(H_test)

V_train = np.concatenate(V_train)
V_test = np.concatenate(V_test)

print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))

def train_and_evaluate(name, y_train, y_test):
    print(f"\nTraining {name} model...")

    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=3,
        max_features=1.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"{name} MAE: {mae:.4f}")
    print(f"{name} RMSE: {rmse:.4f}")
    print(f"{name} R2: {r2:.4f}")

    joblib.dump(model, MODELS / f"{name.lower()}_rf.joblib")

    return {
        "target": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


results = []

results.append(train_and_evaluate("Depth", H_train, H_test))
results.append(train_and_evaluate("Velocity", V_train, V_test))

pd.DataFrame(results).to_csv(
    OUTPUT / "regression_results.csv",
    index=False
)

print("\nBoth regression models trained successfully!")
print("Models saved in:", MODELS)
print("Results saved in:", OUTPUT / "regression_results.csv")
