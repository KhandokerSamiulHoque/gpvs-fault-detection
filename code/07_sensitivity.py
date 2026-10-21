"""Sensitivity of the cross-mode result (E3b) to the estimated onsets, the guard band, and the baseline length.

For each setting we rebuild the onset labels and delta features, then train LightGBM and the random forest
(seed 0) on MPPT runs and test on IPPT runs, and the reverse.
  * onset shift: every estimated onset moved by -1.0, -0.5, -0.25, 0, +0.25, +0.5, +1.0 s
  * guard band: 0.25, 0.5, 1.0 s
  * baseline length: 2, 3, 4 s

Run:  python code/07_sensitivity.py
Writes results/sensitivity.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
RES = Path(__file__).resolve().parents[1] / "results"

w0 = pd.read_csv(RES / "windows.csv")
on = pd.read_csv(RES / "onsets.csv").set_index("file")
META = ["file", "fault", "mode", "win", "t_center"]
RAW = [c for c in w0.columns if c not in META]


def run(shift=0.0, guard=0.5, base_t=3.0):
    w = w0.copy()
    onset = w["file"].map(on["onset_t"]) + shift
    onset[w.fault == 0] = np.inf
    y = np.where(w.t_center >= onset, w.fault, 0)
    near = (w.t_center - onset).abs() < guard
    base = w[w.t_center < base_t].groupby("file")[RAW].mean()
    X = (w[RAW] - base.loc[w["file"]].to_numpy()).to_numpy()
    keep = ((w.t_center >= base_t) & ~near & (w.fault != 4)).to_numpy()
    out = []
    for tr_m, te_m in [("MPPT", "IPPT"), ("IPPT", "MPPT")]:
        tr = keep & (w["mode"] == tr_m).to_numpy()
        te = keep & (w["mode"] == te_m).to_numpy()
        for name, m in [("LightGBM", LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced",
                                                    random_state=0, verbose=-1)),
                        ("RF", RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1,
                                                      random_state=0))]:
            p = m.fit(X[tr], y[tr]).predict(X[te])
            out.append({"direction": f"{tr_m}->{te_m}", "model": name,
                        "macro_f1": f1_score(y[te], p, average="macro")})
    return out


rows = []
for s in [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]:
    rows += [{"factor": "onset_shift_s", "value": s, **r} for r in run(shift=s)]
for g in [0.25, 0.5, 1.0]:
    rows += [{"factor": "guard_s", "value": g, **r} for r in run(guard=g)]
for b in [2.0, 3.0, 4.0]:
    rows += [{"factor": "baseline_s", "value": b, **r} for r in run(base_t=b)]
df = pd.DataFrame(rows)
df.to_csv(RES / "sensitivity.csv", index=False)
print(df.pivot_table(index=["factor", "value"], columns=["model", "direction"], values="macro_f1").round(3).to_string())
