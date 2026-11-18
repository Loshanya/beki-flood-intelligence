
from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# Project paths
ROOT = Path(__file__).resolve().parent / "ML"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
SAMPLES_PER_SCENARIO = 2000
DISCHARGE_CUTOFF = 6000


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
    return data.apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)


def load_scenario(folder, rng):
    q = get_discharge(folder.name)

    h_file = find_file(folder, ["h_avg"])
    v_file = find_file(folder, ["rv_avg"])

    if h_file is None or v_file is None:
        return None

    h = read_grid(h_file)
    v = read_grid(v_file)

    if h.shape != v.shape or h.shape != (286, 600):
        print("Unexpected grid shape:", folder.name)
        return None

    fhi = h * (v + 0.5)

    valid = (
        np.isfinite(h)
        & np.isfinite(v)
        & np.isfinite(fhi)
    )

    indices = np.flatnonzero(valid)

    if len(indices) > SAMPLES_PER_SCENARIO:
        indices = rng.choice(
            indices,
            size=SAMPLES_PER_SCENARIO,
            replace=False
        )

    rows, cols = np.unravel_index(indices, h.shape)

    depth = h.ravel()[indices]
    values = fhi.ravel()[indices]

    wet = depth > 0.1

    labels = np.zeros(len(indices), dtype=np.int8)

    labels[wet & (values <= 0.5)] = 1
    labels[wet & (values > 0.5) & (values <= 1)] = 2
    labels[wet & (values > 1) & (values <= 2)] = 3
    labels[wet & (values > 2) & (values <= 3)] = 4
    labels[wet & (values > 3)] = 5

    X = np.column_stack([
        np.full(len(indices), q, dtype=float),
        rows.astype(float),
        cols.astype(float)
    ])

    return X, labels


def main():
    rng = np.random.default_rng(RANDOM_STATE)

    folders = sorted(
        [
            p for p in ROOT.iterdir()
            if p.is_dir()
            and p.name.startswith("BEKI_FinalQ_")
            and get_discharge(p.name) is not None
        ],
        key=lambda p: get_discharge(p.name)
    )

    train_parts = []
    test_parts = []

    print("Loading hydraulic scenarios...\n")

    for folder in folders:
        q = get_discharge(folder.name)

        result = load_scenario(folder, rng)

        if result is None:
            print(f"Skipping Q={q}")
            continue

        X, y = result

        if q <= DISCHARGE_CUTOFF:
            train_parts.append((X, y))
            print(f"Q={q}: Training data")
        else:
            test_parts.append((X, y))
            print(f"Q={q}: Testing data")

    if not train_parts or not test_parts:
        raise ValueError("Training or testing data is missing.")

    X_train = np.vstack([part[0] for part in train_parts])
    y_train = np.concatenate([part[1] for part in train_parts])

    X_test = np.vstack([part[0] for part in test_parts])
    y_test = np.concatenate([part[1] for part in test_parts])

    print("\n========== DATA SUMMARY ==========")
    print("Training scenarios:", len(train_parts))
    print("Testing scenarios:", len(test_parts))
    print("Training observations:", len(y_train))
    print("Testing observations:", len(y_test))
    print("Training discharge range: 200 to 6000")
    print("Testing discharge range: Above 6000 to 10000")

    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n========== HIGH-DISCHARGE RESULTS ==========")
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2, 3, 4, 5],
            zero_division=0
        )
    )

    report = pd.DataFrame([{
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "training_scenarios": len(train_parts),
        "testing_scenarios": len(test_parts),
        "training_discharge_max": DISCHARGE_CUTOFF,
        "testing_discharge_min": DISCHARGE_CUTOFF + 200
    }])

    report.to_csv(
        OUTPUT / "high_discharge_evaluation.csv",
        index=False
    )

    print("\nResults saved successfully!")
    print(OUTPUT / "high_discharge_evaluation.csv")


if __name__ == "__main__":
    main()
