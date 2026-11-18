
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("ML")

def read_grid(q, filename):
    path = ROOT / f"BEKI_FinalQ_{q}_GVF - Copy" / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_excel(path, header=None).apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

def metrics(actual, predicted, name):
    valid = np.isfinite(actual) & np.isfinite(predicted)
    a, p = actual[valid], predicted[valid]

    mae = np.mean(np.abs(a - p))
    rmse = np.sqrt(np.mean((a - p) ** 2))

    print(f"\n{name}")
    print(f"Comparable cells: {len(a):,}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

def classify(h, v):
    fhi = h * (v + 0.5)
    valid = np.isfinite(h) & np.isfinite(v) & (h > 0.1)

    c = np.zeros(h.shape, dtype=np.uint8)
    c[valid & (fhi <= 0.5)] = 1
    c[valid & (fhi > 0.5) & (fhi <= 1)] = 2
    c[valid & (fhi > 1) & (fhi <= 2)] = 3
    c[valid & (fhi > 2) & (fhi <= 3)] = 4
    c[valid & (fhi > 3)] = 5
    return c

# Estimate Q=400 from Q=200 and Q=600
h200 = read_grid(200, "h_avg.xlsx")
h600 = read_grid(600, "h_avg.xlsx")
v200 = read_grid(200, "rv_avg.xlsx")
v600 = read_grid(600, "rv_avg.xlsx")

h_actual = read_grid(400, "h_avg.xlsx")
v_actual = read_grid(400, "rv_avg.xlsx")

grids = [h200, h600, v200, v600, h_actual, v_actual]
if len({x.shape for x in grids}) != 1:
    raise ValueError("Grid shapes differ; check alignment first.")

h_pred = h200 + (400 - 200) / (600 - 200) * (h600 - h200)
v_pred = v200 + (400 - 200) / (600 - 200) * (v600 - v200)

print("INTERPOLATION VALIDATION: 400 CUMECS")
print("Estimated from 200 and 600; compared with actual 400.")

metrics(h_actual, h_pred, "Water depth")
metrics(v_actual, v_pred, "Velocity")

actual_class = classify(h_actual, v_actual)
predicted_class = classify(h_pred, v_pred)

# Compare cells wet in either map; include dry/wet disagreement
valid = (
    np.isfinite(h_actual) & np.isfinite(v_actual)
    & np.isfinite(h_pred) & np.isfinite(v_pred)
)
agreement = np.mean(
    actual_class[valid] == predicted_class[valid]
) * 100

print(f"\nHazard class agreement: {agreement:.2f}%")
print(f"Cells compared: {valid.sum():,}")
print("\nThis is a validation baseline, not proof of general accuracy.")
