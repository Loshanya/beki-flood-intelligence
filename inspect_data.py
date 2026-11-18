from pathlib import Path
import pandas as pd
import numpy as np

root = Path("ML")

files = {
    1600: ("h_avg.xlsx", "rv_avg.xlsx"),
    1800: ("h_avg.xlsx", "rv_avg.xlsx"),
    2000: ("h_avg_2k.xlsx", "rv_avg_2k.xlsx"),
}

for q, (hfile, vfile) in files.items():
    folder = root / f"BEKI_FinalQ_{q}_GVF - Copy"

    print(f"\nQ = {q}")

    for filename in (hfile, vfile):
        data = pd.read_excel(
            folder / filename, header=None
        ).apply(pd.to_numeric, errors="coerce").to_numpy()

        valid = data[np.isfinite(data)]
        print(
            f"{filename}: mean={valid.mean():.6f}, "
            f"median={np.median(valid):.6f}, "
            f"min={valid.min():.6f}, max={valid.max():.6f}"
        )