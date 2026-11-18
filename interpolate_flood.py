from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("ML")
Q_LOW = 200
Q_HIGH = 400
Q_TARGET = 300

def read_grid(q, filename):
    folder = ROOT / f"BEKI_FinalQ_{q}_GVF - Copy"
    path = folder / filename

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_excel(path, header=None).apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

# Read depth and velocity from both known scenarios
h_low = read_grid(Q_LOW, "h_avg.xlsx")
h_high = read_grid(Q_HIGH, "h_avg.xlsx")

v_low = read_grid(Q_LOW, "rv_avg.xlsx")
v_high = read_grid(Q_HIGH, "rv_avg.xlsx")

# Confirm matching grid shapes
if not (
    h_low.shape == h_high.shape
    == v_low.shape == v_high.shape
):
    raise ValueError("Grid dimensions do not match.")

# Cell-by-cell linear interpolation
weight = (Q_TARGET - Q_LOW) / (Q_HIGH - Q_LOW)

h_target = h_low + weight * (h_high - h_low)
v_target = v_low + weight * (v_high - v_low)

# Provisional FHI, following the existing MATLAB formula
fhi = h_target * (v_target + 0.5)

# Apply the existing script's wet-cell threshold
wet = np.isfinite(h_target) & np.isfinite(v_target) & (h_target > 0.1)

hazard = np.zeros(h_target.shape, dtype=np.uint8)
hazard[wet & (fhi <= 0.5)] = 1
hazard[wet & (fhi > 0.5) & (fhi <= 1.0)] = 2
hazard[wet & (fhi > 1.0) & (fhi <= 2.0)] = 3
hazard[wet & (fhi > 2.0) & (fhi <= 3.0)] = 4
hazard[wet & (fhi > 3.0)] = 5

# Save outputs separately; original files are untouched
out = Path("outputs") / f"Q_{Q_TARGET}"
out.mkdir(parents=True, exist_ok=True)

pd.DataFrame(h_target).to_csv(out / "depth_300.csv", index=False, header=False)
pd.DataFrame(v_target).to_csv(out / "velocity_300.csv", index=False, header=False)
pd.DataFrame(fhi).to_csv(out / "fhi_300.csv", index=False, header=False)
pd.DataFrame(hazard).to_csv(out / "hazard_class_300.csv", index=False, header=False)

print(f"Target discharge: {Q_TARGET} cumecs")
print(f"Grid shape: {h_target.shape}")
print(f"Wet cells: {int(wet.sum())}")
print("Hazard class counts (0=dry/unclassified, 1-5=hazard classes):")

classes, counts = np.unique(hazard, return_counts=True)
for cls, count in zip(classes, counts):
    print(f"  Class {cls}: {count}")

print(f"\nFiles saved to: {out.resolve()}")