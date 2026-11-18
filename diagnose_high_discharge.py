
import os
import glob
import numpy as np
import pandas as pd

from joblib import load
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "ML")
MODEL_DIR = os.path.join(DATA, "models")

wet_bundle = load(
    os.path.join(MODEL_DIR, "wet_classifier.joblib")
)

wet_model = wet_bundle["model"]
threshold = wet_bundle["threshold"]

depth_model = load(
    os.path.join(MODEL_DIR, "depth_rf.joblib")
)

velocity_model = load(
    os.path.join(MODEL_DIR, "velocity_rf.joblib")
)

ROWS, COLS = 286, 600

r, c = np.indices((ROWS, COLS))

def find_folder(q):
    matches = glob.glob(
        os.path.join(DATA, f"BEKI_FinalQ_{q}_GVF*")
    )
    if not matches:
        raise FileNotFoundError(f"Missing Q={q}")
    return matches[0]


def read_grid(folder, prefix):
    matches = glob.glob(
        os.path.join(folder, prefix + "*.xlsx")
    )
    if not matches:
        raise FileNotFoundError(
            f"Missing {prefix} in {folder}"
        )
    return pd.read_excel(
        matches[0], header=None
    ).to_numpy(dtype=float)


for Q in [6200, 8200, 9200]:

    print("\n" + "=" * 55)
    print("DISCHARGE:", Q)
    print("=" * 55)

    folder = find_folder(Q)

    h_actual = read_grid(folder, "h_avg")
    v_actual = read_grid(folder, "rv_avg")

    features = np.column_stack([
        np.full(ROWS * COLS, Q),
        r.ravel(),
        c.ravel()
    ])

    wet_prob = wet_model.predict_proba(
        features
    )[:, 1].reshape(ROWS, COLS)

    wet_pred = wet_prob >= threshold

    h_pred = depth_model.predict(
        features
    ).reshape(ROWS, COLS)

    v_pred = velocity_model.predict(
        features
    ).reshape(ROWS, COLS)

    h_pred = np.maximum(h_pred, 0)
    v_pred = np.maximum(v_pred, 0)

    h_pred[~wet_pred] = 0.1
    v_pred[~wet_pred] = 0

    print("\nWET CELLS")
    print("Actual:", (h_actual > 0.1).sum())
    print("Predicted:", wet_pred.sum())

    print("\nDEPTH STATISTICS")
    print("Actual mean:", h_actual.mean())
    print("Predicted mean:", h_pred.mean())
    print("Actual max:", h_actual.max())
    print("Predicted max:", h_pred.max())

    print("Depth MAE:", mean_absolute_error(
        h_actual.ravel(), h_pred.ravel()
    ))

    print("Depth R2:", r2_score(
        h_actual.ravel(), h_pred.ravel()
    ))

    print("\nVELOCITY STATISTICS")
    print("Actual mean:", v_actual.mean())
    print("Predicted mean:", v_pred.mean())
    print("Actual max:", v_actual.max())
    print("Predicted max:", v_pred.max())

    actual_wet = h_actual > 0.1

    print(
        "Wet-region depth MAE:",
        mean_absolute_error(
            h_actual[actual_wet],
            h_pred[actual_wet]
        )
    )

    print(
        "Wet-region velocity MAE:",
        mean_absolute_error(
            v_actual[actual_wet],
            v_pred[actual_wet]
        )
    )

    print(
        "Depth R2 on wet cells:",
        r2_score(
            h_actual[actual_wet],
            h_pred[actual_wet]
        )
    )

    print(
        "Velocity R2 on wet cells:",
        r2_score(
            v_actual[actual_wet],
            v_pred[actual_wet]
        )
    )

print("\nHIGH-DISCHARGE DIAGNOSTICS COMPLETED")
