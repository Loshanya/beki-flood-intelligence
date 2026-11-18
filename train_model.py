
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "ML"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

def get_discharge(folder_name):
    match = re.search(r"BEKI_FinalQ_(\d+)_", folder_name)
    return int(match.group(1)) if match else None

def find_file(folder, prefixes):
    for prefix in prefixes:
        matches = [
            f for f in folder.glob("*.xlsx")
            if f.name.lower().startswith(prefix)
            and "_org" not in f.name.lower()
        ]
        if matches:
            return matches[0]
    return None

def read_grid(path):
    data = pd.read_excel(path, header=None)
    return data.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

folders = sorted(
    [
        p for p in ROOT.iterdir()
        if p.is_dir() and p.name.startswith("BEKI_FinalQ_")
    ],
    key=lambda p: get_discharge(p.name) or 0
)

summary = []
usable = []

print("Checking flood scenarios...\n")

for folder in folders:
    q = get_discharge(folder.name)
    h_file = find_file(folder, ["h_avg"])
    v_file = find_file(folder, ["rv_avg"])

    if h_file is None or v_file is None:
        summary.append({
            "discharge": q,
            "status": "Missing average depth or velocity file"
        })
        continue

    try:
        h = read_grid(h_file)
        v = read_grid(v_file)

        if h.shape != v.shape or h.shape != (286, 600):
            summary.append({
                "discharge": q,
                "status": f"Unexpected grid shapes: h={h.shape}, v={v.shape}"
            })
            continue

        valid = np.isfinite(h) & np.isfinite(v)
        if valid.sum() == 0:
            summary.append({
                "discharge": q,
                "status": "No valid cells"
            })
            continue

        # Provisional labels based on the existing MATLAB FHI formula.
        fhi = h * (v + 0.5)
        labels = np.zeros(h.shape, dtype=np.int8)
        wet = h > 0.1

        labels[wet & (fhi <= 0.5)] = 1
        labels[wet & (fhi > 0.5) & (fhi <= 1.0)] = 2
        labels[wet & (fhi > 1.0) & (fhi <= 2.0)] = 3
        labels[wet & (fhi > 2.0) & (fhi <= 3.0)] = 4
        labels[wet & (fhi > 3.0)] = 5

        valid &= np.isfinite(fhi)

        usable.append({
            "discharge": q,
            "folder": folder.name,
            "valid_cells": int(valid.sum()),
            "class_counts": np.bincount(
                labels[valid], minlength=6
            ).tolist()
        })

        summary.append({
            "discharge": q,
            "status": "OK",
            "valid_cells": int(valid.sum()),
            "depth_file": h_file.name,
            "velocity_file": v_file.name
        })

    except Exception as exc:
        summary.append({
            "discharge": q,
            "status": f"Read error: {exc}"
        })

report = pd.DataFrame(summary).sort_values("discharge")
report.to_csv(OUTPUT / "scenario_check.csv", index=False)

print(report.to_string(index=False))
print(f"\nUsable scenarios: {len(usable)} of {len(folders)}")
print("Report saved to:", OUTPUT / "scenario_check.csv")
print("\nNo model has been trained yet.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

print("\nPreparing a sampled ML dataset...")

rng = np.random.default_rng(42)
X_parts = []
y_parts = []

# Sample cells from each usable scenario to limit memory usage.
for item in usable:
    q = item["discharge"]
    folder = ROOT / item["folder"]

    h_file = find_file(folder, ["h_avg"])
    v_file = find_file(folder, ["rv_avg"])

    h = read_grid(h_file)
    v = read_grid(v_file)

    fhi = h * (v + 0.5)
    labels = np.zeros(h.shape, dtype=np.int8)
    wet = h > 0.1

    labels[wet & (fhi <= 0.5)] = 1
    labels[wet & (fhi > 0.5) & (fhi <= 1.0)] = 2
    labels[wet & (fhi > 1.0) & (fhi <= 2.0)] = 3
    labels[wet & (fhi > 2.0) & (fhi <= 3.0)] = 4
    labels[wet & (fhi > 3.0)] = 5

    valid = np.isfinite(h) & np.isfinite(v) & np.isfinite(fhi)
    indices = np.flatnonzero(valid)

    # Up to 2,000 cells per scenario.
    if len(indices) > 2000:
        indices = rng.choice(indices, size=2000, replace=False)

    rows, cols = np.unravel_index(indices, h.shape)

    X_parts.append(np.column_stack([
        np.full(len(indices), q, dtype=float),
        rows.astype(float),
        cols.astype(float)
    ]))
    y_parts.append(labels.ravel()[indices])

X = np.vstack(X_parts)
y = np.concatenate(y_parts)

print("Sampled rows:", len(y))
print("Features: discharge, row, column")
print("Classes:", np.unique(y, return_counts=True))

# Random split is a quick prototype test, not an independent
# held-out-discharge evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nPrototype evaluation:")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Macro F1:", round(f1_score(y_test, pred, average="macro", zero_division=0), 4))
print("\nClassification report:")
print(classification_report(y_test, pred, zero_division=0))

joblib.dump(model, OUTPUT / "flood_hazard_rf.joblib")
print("Model saved to:", OUTPUT / "flood_hazard_rf.joblib")
print("Reminder: random-split scores are not held-out-discharge validation.")
