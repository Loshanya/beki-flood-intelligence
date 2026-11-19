# Leave-one-scenario-out test of hydraulic interpolation.
# Each discharge is rebuilt from the two scenarios gap m3/s either side
# and compared with the simulated grid. Run from the repo root.
import glob, re
import numpy as np
import pandas as pd

ROOT = "ML"
GAP_LIST = [200, 400]
SKIP = {1800, 3600, 4800, 5600}   # files that look duplicated or derived, see report ch. 4


def load(q):
    f = glob.glob(f"{ROOT}/BEKI_FinalQ_{q}_GVF*")[0]
    h = pd.read_excel(glob.glob(f + "/h_avg*.xlsx")[0], header=None).to_numpy(float)
    v = pd.read_excel(glob.glob(f + "/rv_avg*.xlsx")[0], header=None).to_numpy(float)
    return h, v


def classify(h, v):
    fhi = h * (v + 0.5)
    wet = h > 0.1
    c = np.zeros(h.shape, int)
    c[wet & (fhi <= 0.5)] = 1
    c[wet & (fhi > 0.5) & (fhi <= 1)] = 2
    c[wet & (fhi > 1) & (fhi <= 2)] = 3
    c[wet & (fhi > 2) & (fhi <= 3)] = 4
    c[wet & (fhi > 3)] = 5
    return fhi, c, wet


qs = sorted(int(re.search(r"FinalQ_(\d+)", p).group(1)) for p in glob.glob(f"{ROOT}/BEKI_FinalQ_*_GVF*"))
rows = []
for gap in GAP_LIST:
    for q in qs:
        a, b = q - gap, q + gap
        if a not in qs or b not in qs or {q, a, b} & SKIP:
            continue
        h, v = load(q)
        ha, va = load(a)
        hb, vb = load(b)
        w = (q - a) / (b - a)
        hp, vp = ha + w * (hb - ha), va + w * (vb - va)
        f, c, wet = classify(h, v)
        fp, cp, wetp = classify(hp, vp)
        rows.append(dict(gap=gap, Q=q,
                         depth_mae=np.abs(h - hp).mean(),
                         vel_mae=np.abs(v - vp).mean(),
                         fhi_mae=np.abs(f - fp).mean(),
                         wet_acc=(wet == wetp).mean() * 100,
                         haz_acc=(c == cp).mean() * 100))

out = pd.DataFrame(rows)
out.to_csv("ML/outputs/interpolation_loo_validation.csv", index=False)
print(out.groupby("gap").mean(numeric_only=True).round(3))
