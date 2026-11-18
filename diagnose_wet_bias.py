
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "ML")
OUT = os.path.join(DATA, "outputs")

Q = 5350

def find_folder(q):
    matches = glob.glob(os.path.join(DATA, f"BEKI_FinalQ_{q}_GVF*"))
    if not matches:
        raise FileNotFoundError(f"Missing scenario Q={q}")
    return matches[0]

def read_grid(folder, prefix):
    matches = glob.glob(os.path.join(folder, prefix + "*.xlsx"))
    if not matches:
        raise FileNotFoundError(f"Missing {prefix} in {folder}")
    return pd.read_excel(matches[0], header=None).to_numpy(dtype=float)

print("Loading hydraulic reference...")

h1 = read_grid(find_folder(5200), "h_avg")
h2 = read_grid(find_folder(5400), "h_avg")
v1 = read_grid(find_folder(5200), "rv_avg")
v2 = read_grid(find_folder(5400), "rv_avg")

# Interpolate to Q=5350
w = (Q - 5200) / (5400 - 5200)
h_hyd = h1 * (1-w) + h2 * w
v_hyd = v1 * (1-w) + v2 * w

ml = np.load(os.path.join(OUT, "ml_prediction_Q5350.npz"))
h_ml = ml["depth"]
v_ml = ml["velocity"]

if h_hyd.shape != h_ml.shape:
    raise ValueError("Grid dimensions do not match.")

# Wet/dry masks
hyd_wet = h_hyd > 0.1
ml_wet = h_ml > 0.1

false_wet = (~hyd_wet) & ml_wet
missed_wet = hyd_wet & (~ml_wet)
correct_wet = hyd_wet & ml_wet
correct_dry = (~hyd_wet) & (~ml_wet)

print("\n========== WET/DRY DIAGNOSTICS ==========")
print("False wet predictions:", false_wet.sum())
print("Missed wet cells:", missed_wet.sum())
print("Correctly predicted wet:", correct_wet.sum())
print("Correctly predicted dry:", correct_dry.sum())

print("\n========== DEPTH ERROR BREAKDOWN ==========")

def report(name, mask):
    if mask.sum() == 0:
        print(name, ": No cells")
        return

    actual = h_hyd[mask]
    predicted = h_ml[mask]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print(f"\n{name}: {mask.sum()} cells")
    print(f"Hydraulic mean depth: {actual.mean():.4f} m")
    print(f"ML mean depth: {predicted.mean():.4f} m")
    print(f"MAE: {mae:.4f} m")
    print(f"RMSE: {rmse:.4f} m")

report("Hydraulically dry cells", ~hyd_wet)
report("Hydraulically wet cells", hyd_wet)
report("False wet predictions", false_wet)
report("Correctly predicted wet cells", correct_wet)

print("\n========== VELOCITY ERROR ON WET CELLS ==========")
print("Hydraulic mean velocity:", v_hyd[hyd_wet].mean())
print("ML mean velocity:", v_ml[hyd_wet].mean())
print(
    "Velocity MAE:",
    mean_absolute_error(v_hyd[hyd_wet], v_ml[hyd_wet])
)

# Save cell-level diagnostic summary
summary = pd.DataFrame([
    ["Hydraulic wet", int(hyd_wet.sum())],
    ["ML wet", int(ml_wet.sum())],
    ["False wet", int(false_wet.sum())],
    ["Missed wet", int(missed_wet.sum())],
    ["Correct wet", int(correct_wet.sum())],
    ["Correct dry", int(correct_dry.sum())]
], columns=["Category", "Cell_Count"])

summary.to_csv(
    os.path.join(OUT, "wet_dry_diagnostics_Q5350.csv"),
    index=False
)

print("\nDiagnostics saved successfully.")
