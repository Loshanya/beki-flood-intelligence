
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

ROOT = Path(__file__).resolve().parent / "ML"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
SAMPLES_PER_SCENARIO = 2000

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

    wet = h.ravel()[indices] > 0.1
    values = fhi.ravel()[indices]

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

    print("Loading scenarios...")

    scenarios = []

    for folder in folders:
        result = load_scenario(folder, rng)

        if result is None:
            print("Skipping:", folder.name)
            continue

        X, y = result

        scenarios.append({
            "discharge": get_discharge(folder.name),
            "X": X,
            "y": y
        })

        print(
            f"Loaded Q={get_discharge(folder.name)}: "
            f"{len(y)} observations"
        )

    if len(scenarios) < 2:
        raise ValueError("Not enough usable scenarios.")

    # Hold out every fifth scenario in discharge order.
    # This evaluates unseen discharge scenarios.
    test_indices = np.arange(len(scenarios)) % 5 == 0

    train_scenarios = [
        s for i, s in enumerate(scenarios)
        if not test_indices[i]
    ]

    test_scenarios = [
        s for i, s in enumerate(scenarios)
        if test_indices[i]
    ]

    X_train = np.vstack([s["X"] for s in train_scenarios])
    y_train = np.concatenate([s["y"] for s in train_scenarios])

    X_test = np.vstack([s["X"] for s in test_scenarios])
    y_test = np.concatenate([s["y"] for s in test_scenarios])

    print("\n========== DATA SPLIT ==========")
    print("Training scenarios:", len(train_scenarios))
    print("Testing scenarios:", len(test_scenarios))
    print(
        "Held-out discharges:",
        [s["discharge"] for s in test_scenarios]
    )
    print("Training observations:", len(y_train))
    print("Testing observations:", len(y_test))

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

    print("\n========== UNSEEN SCENARIO RESULTS ==========")
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
        "training_scenarios": len(train_scenarios),
        "testing_scenarios": len(test_scenarios),
        "testing_discharges": ", ".join(
            str(s["discharge"]) for s in test_scenarios
        )
    }])

    report.to_csv(
        OUTPUT / "unseen_scenario_evaluation.csv",
        index=False
    )

    print(
        "\nResults saved to:",
        OUTPUT / "unseen_scenario_evaluation.csv"
    )


if __name__ == "__main__":
    main()
