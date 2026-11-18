
import os
import glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "ML")
OUTPUT = os.path.join(DATA, "outputs")

os.makedirs(OUTPUT, exist_ok=True)

for Q in [6200, 8200, 9200]:

    folder_matches = glob.glob(
        os.path.join(DATA, f"BEKI_FinalQ_{Q}_GVF*")
    )

    if not folder_matches:
        print("Missing scenario:", Q)
        continue

    folder = folder_matches[0]

    def read_grid(prefix):
        files = glob.glob(
            os.path.join(folder, prefix + "*.xlsx")
        )

        if not files:
            raise FileNotFoundError(prefix)

        return pd.read_excel(
            files[0], header=None
        ).to_numpy(dtype=float)

    h = read_grid("h_avg")
    v = read_grid("rv_avg")

    wet = h > 0.1
    fhi = h * (v + 0.5)

    print("\n" + "=" * 55)
    print("DISCHARGE:", Q)
    print("=" * 55)

    for name, arr, thresholds in [
        ("DEPTH", h, [1, 2, 5, 10, 20, 40]),
        ("VELOCITY", v, [0.5, 1, 2, 5, 10, 20, 40]),
        ("FHI", fhi, [0.5, 1, 2, 3, 5, 10, 20])
    ]:

        print("\n", name)

        values = arr[wet]

        for t in thresholds:
            print(
                f"Cells above {t}:",
                int(np.sum(values > t))
            )

        print("Mean:", values.mean())
        print("Median:", np.median(values))
        print("95th percentile:", np.percentile(values, 95))
        print("99th percentile:", np.percentile(values, 99))
        print("Maximum:", values.max())

    # Identify top 20 velocity cells
    indices = np.argwhere(wet)

    wet_velocity = v[wet]
    top_indices = np.argsort(wet_velocity)[-20:][::-1]

    records = []

    for idx in top_indices:
        row, col = indices[idx]

        records.append({
            "Discharge": Q,
            "Row": int(row),
            "Column": int(col),
            "Depth": float(h[row, col]),
            "Velocity": float(v[row, col]),
            "FHI": float(fhi[row, col])
        })

    pd.DataFrame(records).to_csv(
        os.path.join(
            OUTPUT,
            f"extreme_velocity_cells_Q{Q}.csv"
        ),
        index=False
    )

print("\nExtreme-cell analysis completed.")
