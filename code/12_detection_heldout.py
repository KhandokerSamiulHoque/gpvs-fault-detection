"""E4 detection with a held-out threshold.

Same two detectors as 04_experiments.py (static Hotelling T^2 and sliding baseline), but the alarm threshold is the
99.5th percentile of the score in the fault-free run of the same mode between 4 s and 9 s only. It is locked before
any fault run is scored. False alarms are then reported on the fault-free run from 9 s to the end (a held-out
temporal test from the same run, not a separate run) and on every fault run's own pre-onset windows.

The first version (threshold from the whole fault-free run) is kept as detection_results_v1.csv.

Run:  python code/12_detection_heldout.py
Writes results/detection_results.csv and updates the thresholds in t2_traces.json
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
RES = Path(__file__).resolve().parents[1] / "results"
BASE_T, GUARD_T, W_PER_S = 3.0, 0.5, 50
CAL_LO, CAL_HI = 4.0, 9.0   # calibration window on the fault-free run (s)

if not (RES / "detection_results_v1.csv").exists():
    shutil.copy(RES / "detection_results.csv", RES / "detection_results_v1.csv")

w = pd.read_csv(RES / "windows.csv")
on = pd.read_csv(RES / "onsets.csv").set_index("file")
META = ["file", "fault", "mode", "win", "t_center"]
RAW_FEATS = [c for c in w.columns if c not in META]
base = w[w.t_center < BASE_T].groupby("file")[RAW_FEATS].mean()
w[[f"d_{c}" for c in RAW_FEATS]] = (w[RAW_FEATS] - base.loc[w["file"]].to_numpy()).to_numpy()

DET = ["Ipv_mean", "Vpv_mean", "Vdc_mean", "Iabc_mean", "Ipv_std", "Vdc_std",
       "ia_std", "ib_std", "ic_std", "phys_I_unbalance", "phys_Pdc", "phys_Vdc_ripple"]
DET_FEATS = [f"d_{c}" for c in DET]


def static_scores(dm):
    """Hotelling T^2 of delta features against the pooled first-3-s windows of this mode, 1 s median."""
    basewin = dm[dm.t_center < BASE_T][DET_FEATS].to_numpy()
    mu = basewin.mean(0)
    inv = np.linalg.pinv(np.cov(basewin, rowvar=False) + 1e-9 * np.eye(len(DET_FEATS)))
    z = dm[DET_FEATS].to_numpy() - mu
    t2 = np.einsum("ij,jk,ik->i", z, inv, z)
    return pd.Series(t2, index=dm.index).groupby(dm["file"]).transform(
        lambda s: s.rolling(W_PER_S, min_periods=1).median())


def sliding_scores(dm):
    """Mean squared z of raw features against the same run's windows from t-4 s to t-1 s, 0.5 s median."""
    lag_lo, lag_hi = 4 * W_PER_S, 1 * W_PER_S
    sd = dm[dm.t_center < BASE_T].groupby("file")[DET].std().median().to_numpy() + 1e-9
    out = pd.Series(np.nan, index=dm.index)
    for f, g in dm.groupby("file"):
        X = g[DET].to_numpy()
        csum = np.vstack([np.zeros(len(DET)), np.cumsum(X, 0)])
        s = np.full(len(X), np.nan)
        for k in range(lag_lo, len(X)):
            ref = (csum[k - lag_hi] - csum[k - lag_lo]) / (lag_lo - lag_hi)
            s[k] = np.mean(((X[k] - ref) / sd) ** 2)
        out.loc[g.index] = pd.Series(s).rolling(W_PER_S // 2, min_periods=1).median().to_numpy()
    return out


traces = json.loads((RES / "t2_traces.json").read_text())
rows = []
for detector, fn, start in [("static", static_scores, BASE_T), ("sliding", sliding_scores, BASE_T + 1)]:
    for mode in ["MPPT", "IPPT"]:
        dm = w[w["mode"] == mode]
        s_all = fn(dm)
        f0 = dm[dm.fault == 0]
        cal = f0[(f0.t_center >= CAL_LO) & (f0.t_center < CAL_HI)].index
        thr = float(np.nanpercentile(s_all.loc[cal], 99.5))      # locked before scoring fault runs
        for f, g in dm.groupby("file"):
            s = s_all.loc[g.index]
            key = "thr" if detector == "static" else "slide_thr"
            traces[f][key] = thr
            is_fault = g.fault.iloc[0] > 0
            onset = on.loc[f, "onset_t"]
            if is_fault:
                mask = (g.t_center >= start) & (g.t_center < onset - GUARD_T)
            else:
                mask = g.t_center >= CAL_HI                       # held-out part of the fault-free run
            r = {"detector": detector, "file": f, "mode": mode, "fault": int(g.fault.iloc[0]), "threshold": thr,
                 "false_alarm_rate": float((s[mask] > thr).mean()),
                 "false_alarm_basis": "pre-onset" if is_fault else "F0 held-out 9 s to end",
                 "onset_t": float(onset) if is_fault else None}
            if is_fault:
                post = g.t_center >= onset
                alarm = s[post] > thr
                if alarm.any():
                    r.update({"detected": True, "delay_s": float(g.t_center[post][alarm].iloc[0]) - onset,
                              "post_alarm_fraction": float(alarm.mean())})
                else:
                    r.update({"detected": False, "delay_s": None, "post_alarm_fraction": 0.0})
            rows.append(r)

det = pd.DataFrame(rows)
det.to_csv(RES / "detection_results.csv", index=False)
(RES / "t2_traces.json").write_text(json.dumps(traces))
print(det.round(3).to_string())
