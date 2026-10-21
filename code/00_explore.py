"""Quick look at the GPVS-Faults CSV files: shape, time span, sampling, and per-column ranges.

Run:  python code/00_explore.py
Writes results/explore_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "gpvs_csv" / "CSV_Files"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for f in sorted(RAW.glob("F*.csv")):
    d = pd.read_csv(f)
    t = d["Time"].to_numpy()
    dt = np.diff(t)
    rows.append({
        "file": f.stem,
        "n_rows": len(d),
        "t_start": t[0],
        "t_end": t[-1],
        "dt_median_us": np.median(dt) * 1e6,
        "dt_max_ms": dt.max() * 1e3,
        "n_nan": int(d.isna().sum().sum()),
        **{f"{c}_mean": d[c].mean() for c in d.columns if c != "Time"},
    })
    if f.stem == "F0M":
        print("columns:", list(d.columns))
        print(d.describe().T.round(3).to_string())

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "explore_summary.csv", index=False)
print(summary[["file", "n_rows", "t_start", "t_end", "dt_median_us", "dt_max_ms", "n_nan"]].to_string())
