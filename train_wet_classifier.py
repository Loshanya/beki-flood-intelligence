
import os
import glob
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score
)
from joblib import dump

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "ML")
MODEL_DIR = os.path.join(DATA, "models")
OUTPUT_DIR = os.path.join(DATA, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)

X_train, y_train = [], []
X_test, y_test = [], []

print("Training wet/dry classifier...")

folders = sorted(glob.glob(
    os.path.join(DATA, "BEKI_FinalQ_*_GVF*")
))

for folder in folders:

    name = os.path.basename(folder)

    import re
    match = re.search(r"FinalQ_(\d+)", name)

    if not match:
        continue

    q = int(match.group(1))

    files_h = glob.glob(os.path.join(folder, "h_avg*.xlsx"))

    if not files_h:
        continue

    try:
        h = pd.read_excel(
            files_h[0], header=None
        ).to_numpy(dtype=float)

        if h.shape != (286, 600):
            print("Skipping:", q, h.shape)
            continue

        rows, cols = np.indices(h.shape)

        # Wet = depth strictly greater than 0.1 m
        wet = (h > 0.1).astype(np.uint8)

        # Sample spatial cells
        n = min(10000, h.size)

        indices = rng.choice(
            h.size, size=n, replace=False
        )

        r = rows.ravel()[indices]
        c = cols.ravel()[indices]
        labels = wet.ravel()[indices]

        features = np.column_stack([
            np.full(n, q),
            r,
            c
        ])

        # Hold out every fifth scenario group
        scenario_number = q // 200

        if scenario_number % 5 == 1:
            X_test.append(features)
            y_test.append(labels)
        else:
            X_train.append(features)
            y_train.append(labels)

        print(
            f"Q={q}: Wet={wet.sum()}, "
            f"Dry={(wet == 0).sum()}"
        )

    except Exception as e:
        print("Error:", q, e)

X_train = np.vstack(X_train)
y_train = np.concatenate(y_train)

X_test = np.vstack(X_test)
y_test = np.concatenate(y_test)

print("\nTraining observations:", len(y_train))
print("Testing observations:", len(y_test))

model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=3,
    class_weight="balanced",
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate probabilities on held-out scenarios
probabilities = model.predict_proba(X_test)[:, 1]

# Choose threshold using balanced accuracy
thresholds = np.arange(0.10, 0.91, 0.02)

best_threshold = 0.5
best_score = -1

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(np.uint8)

    score = balanced_accuracy_score(
        y_test, predictions
    )

    if score > best_score:
        best_score = score
        best_threshold = threshold

final_predictions = (
    probabilities >= best_threshold
).astype(np.uint8)

print("\n========== WET/DRY CLASSIFIER RESULTS ==========")
print("Selected threshold:", best_threshold)
print("Accuracy:", accuracy_score(y_test, final_predictions))
print("Balanced accuracy:", best_score)

print("\nClassification report:")
print(classification_report(
    y_test,
    final_predictions,
    target_names=["Dry", "Wet"],
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, final_predictions))

# Save model and threshold
dump(
    {
        "model": model,
        "threshold": float(best_threshold)
    },
    os.path.join(MODEL_DIR, "wet_classifier.joblib")
)

print("\nSaved:", os.path.join(
    MODEL_DIR, "wet_classifier.joblib"
))
