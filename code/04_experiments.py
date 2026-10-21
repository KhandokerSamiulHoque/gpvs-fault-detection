"""Main experiments for Paper B.

E1  Reproduce the common protocol: every window gets its run's fault label, random 70/30 window split.
E2  Shortcut probe: same protocol but only windows BEFORE the fault onset (physically fault-free).
    If classes are still separable, the model is recognizing the run, not the fault.
E3  Leakage-aware fault classification with onset-based labels
    (pre-onset = normal, post-onset = fault type; F4 excluded because no onset can be located):
      a) random window split            (optimistic reference)
      b) cross-mode split: train on MPPT runs, test on IPPT runs, and the reverse
    each with RAW features and BASELINE-REFERENCED (delta) features, five model families.
E4  Unsupervised detection: Hotelling T^2 on baseline-referenced features, threshold set on the
    fault-free runs; detection rate, delay after onset, and false alarms before onset.

Run:  python code/04_experiments.py
Writes results/*.csv, *.json and predictions used by 05_figures.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

SEEDS = [0, 1, 2, 3, 4]
BASE_T = 3.0      # seconds of each run used as its fault-free baseline
GUARD_T = 0.5     # windows within +/- this many seconds of the onset are dropped
W_PER_S = 50

w = pd.read_csv(RES / "windows.csv")
on = pd.read_csv(RES / "onsets.csv").set_index("file")
META = ["file", "fault", "mode", "win", "t_center"]
RAW_FEATS = [c for c in w.columns if c not in META]
STAT_FEATS = [c for c in RAW_FEATS if not c.startswith("phys_")]

# ---------- labels ----------
w["onset_t"] = w["file"].map(on["onset_t"])
w["locatable"] = w["file"].map(on["locatable"])
w.loc[w.fault == 0, "onset_t"] = np.inf
w["post"] = w["t_center"] >= w["onset_t"]
w["near_onset"] = (w["t_center"] - w["onset_t"]).abs() < GUARD_T
w["y_onset"] = np.where(w["post"], w["fault"], 0)

# ---------- baseline-referenced (delta) features ----------
base = w[w.t_center < BASE_T].groupby("file")[RAW_FEATS].mean()
delta = w[RAW_FEATS] - base.loc[w["file"]].to_numpy()
DELTA_FEATS = [f"d_{c}" for c in RAW_FEATS]
w[DELTA_FEATS] = delta.to_numpy()


def models(seed):
    return {
        "RF": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced",
                                   random_state=seed, verbose=-1),
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced",
                                                                     random_state=seed)),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced",
                                                       random_state=seed)),
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


def scores(y, p):
    return {"acc": float(np.mean(y == p)), "bal_acc": float(balanced_accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro"))}


def block_bootstrap_ci(y, p, files, n_boot=1000, block=W_PER_S, seed=0):
    """95% CI for macro-F1 by resampling contiguous 1 s blocks of test windows within each run."""
    rng = np.random.default_rng(seed)
    blocks = []
    for f in np.unique(files):
        idx = np.where(files == f)[0]
        blocks += [idx[i:i + block] for i in range(0, len(idx), block)]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        ii = np.concatenate([blocks[k] for k in pick])
        vals.append(f1_score(y[ii], p[ii], average="macro"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


rows, preds = [], {}

# ================= E1: file labels, random split =================
for scope in ["MPPT", "both"]:
    d = w if scope == "both" else w[w["mode"] == "MPPT"]
    X, y = d[STAT_FEATS].to_numpy(), d["fault"].to_numpy()
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
        rows.append({"exp": "E1_file_label_random", "scope": scope, "model": "RF", "features": "stat52",
                     "seed": seed, **scores(yte, m.predict(Xte))})

# ================= E2: shortcut probe on fault-free windows =================
# Only windows before onset (and outside the guard band) of runs with a locatable onset, plus F0 runs
# truncated to the same median time span. Label = the run's fault id. Physically all are normal.
pre = w[(~w.post) & (~w.near_onset) & (w.locatable | (w.fault == 0))]
for scope in ["MPPT", "IPPT"]:
    d = pre[pre["mode"] == scope]
    X, y = d[STAT_FEATS].to_numpy(), d["fault"].to_numpy()
    n_cls = len(np.unique(y))
    for seed in SEEDS:
        # random split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
        rows.append({"exp": "E2_prefault_random", "scope": scope, "model": "RF", "features": "stat52",
                     "seed": seed, "n_classes": n_cls, "chance": 1 / n_cls, **scores(yte, m.predict(Xte))})
        # time-blocked split: first 70% of each run's pre-fault windows train, last 30% test
        tr_idx, te_idx = [], []
        for f, g in d.groupby("file"):
            cut = int(len(g) * 0.7)
            tr_idx += list(g.index[:cut])
            te_idx += list(g.index[cut:])
        m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed).fit(
            d.loc[tr_idx, STAT_FEATS], d.loc[tr_idx, "fault"])
        rows.append({"exp": "E2_prefault_blocked", "scope": scope, "model": "RF", "features": "stat52",
                     "seed": seed, "n_classes": n_cls, "chance": 1 / n_cls,
                     **scores(d.loc[te_idx, "fault"].to_numpy(), m.predict(d.loc[te_idx, STAT_FEATS]))})

# ================= E3: onset labels =================
e3 = w[(w.t_center >= BASE_T) & (~w.near_onset) & (w.fault != 4)].copy()
FEATSETS = {"raw": RAW_FEATS, "delta": DELTA_FEATS}
class_names = sorted(e3.y_onset.unique())

for fs_name, feats in FEATSETS.items():
    X, y = e3[feats].to_numpy(), e3["y_onset"].to_numpy()
    # a) random split
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        for mname, m in models(seed).items():
            p = m.fit(Xtr, ytr).predict(Xte)
            rows.append({"exp": "E3a_onset_random", "scope": "both", "model": mname, "features": fs_name,
                         "seed": seed, **scores(yte, p)})
            if seed == 0:
                preds[f"E3a|{fs_name}|{mname}"] = {"y": yte.tolist(), "p": p.tolist()}
    # b) cross-mode
    for tr_mode, te_mode in [("MPPT", "IPPT"), ("IPPT", "MPPT")]:
        tr, te = e3[e3["mode"] == tr_mode], e3[e3["mode"] == te_mode]
        for seed in SEEDS:
            for mname, m in models(seed).items():
                p = m.fit(tr[feats].to_numpy(), tr["y_onset"].to_numpy()).predict(te[feats].to_numpy())
                yte = te["y_onset"].to_numpy()
                r = {"exp": "E3b_cross_mode", "scope": f"{tr_mode}->{te_mode}", "model": mname,
                     "features": fs_name, "seed": seed, **scores(yte, p)}
                rec = recall_score(yte, p, labels=class_names, average=None, zero_division=0)
                r.update({f"recall_F{c}": float(v) for c, v in zip(class_names, rec)})
                if seed == 0:
                    lo_ci, hi_ci = block_bootstrap_ci(yte, p, te["file"].to_numpy())
                    r.update({"f1_ci_lo": lo_ci, "f1_ci_hi": hi_ci})
                    preds[f"E3b|{fs_name}|{mname}|{tr_mode}->{te_mode}"] = {"y": yte.tolist(), "p": p.tolist()}
                rows.append(r)
    print("E3 done for", fs_name)

res = pd.DataFrame(rows)
res.to_csv(RES / "experiment_results.csv", index=False)

# ================= E4: unsupervised detection =================
# Per run: mean and covariance of baseline-referenced features over the first BASE_T seconds are not
# usable (deltas are ~0 by construction), so we pool the baseline windows of all runs of the same mode
# to estimate a covariance, then score every window of every run with Hotelling T^2.
DET_FEATS = [f"d_{c}" for c in ["Ipv_mean", "Vpv_mean", "Vdc_mean", "Iabc_mean", "Ipv_std", "Vdc_std",
                                "ia_std", "ib_std", "ic_std", "phys_I_unbalance", "phys_Pdc", "phys_Vdc_ripple"]]
det_rows, t2_trace = [], {}
for mode in ["MPPT", "IPPT"]:
    dm = w[w["mode"] == mode]
    basewin = dm[dm.t_center < BASE_T][DET_FEATS].to_numpy()
    mu = basewin.mean(0)
    cov = np.cov(basewin, rowvar=False) + 1e-9 * np.eye(len(DET_FEATS))
    inv = np.linalg.pinv(cov)
    z = dm[DET_FEATS].to_numpy() - mu
    t2 = np.einsum("ij,jk,ik->i", z, inv, z)
    # smooth over 1 s (median) so a single noisy cycle does not raise an alarm
    t2s = pd.Series(t2, index=dm.index).groupby(dm["file"]).transform(
        lambda s: s.rolling(W_PER_S, min_periods=1).median())
    # threshold: 99.5th percentile of smoothed T^2 over the full fault-free run of this mode (after BASE_T)
    f0 = dm[(dm.fault == 0) & (dm.t_center >= BASE_T)].index
    thr = float(np.percentile(t2s.loc[f0], 99.5))
    for f, g in dm.groupby("file"):
        s = t2s.loc[g.index]
        t2_trace[f] = {"t": g.t_center.round(4).tolist(), "t2": s.round(4).tolist(), "thr": thr}
        after_base = g.t_center >= BASE_T
        onset = on.loc[f, "onset_t"]
        pre_mask = after_base & (g.t_center < onset - GUARD_T) if g.fault.iloc[0] > 0 else after_base
        fa_rate = float((s[pre_mask] > thr).mean())
        r = {"file": f, "mode": mode, "fault": int(g.fault.iloc[0]), "threshold": thr,
             "false_alarm_rate_pre": fa_rate, "onset_t": float(onset) if g.fault.iloc[0] > 0 else None}
        if g.fault.iloc[0] > 0:
            post = g.t_center >= onset
            alarm = s[post] > thr
            if alarm.any():
                t_alarm = float(g.t_center[post][alarm].iloc[0])
                r.update({"detected": True, "delay_s": t_alarm - onset,
                          "post_alarm_fraction": float(alarm.mean())})
            else:
                r.update({"detected": False, "delay_s": None, "post_alarm_fraction": 0.0})
        det_rows.append(r)
det = pd.DataFrame(det_rows)
det["detector"] = "static"

# ---------- E4b: sliding-baseline detector ----------
# The static baseline ages: slow irradiance/temperature drift moves the operating point, so T^2 climbs
# even in fault-free runs. Here each window is compared with the same run's own recent past,
# windows from t-4 s to t-1 s (the 1 s gap keeps a fresh fault out of its own reference).
# Score = mean over features of z^2, z = (x - lagged mean) / pooled baseline sd of that feature.
LAG_LO, LAG_HI = 4 * W_PER_S, 1 * W_PER_S
RAW_DET = [c[2:] for c in DET_FEATS]
det2_rows = []
for mode in ["MPPT", "IPPT"]:
    dm = w[w["mode"] == mode]
    sd = dm[dm.t_center < BASE_T].groupby("file")[RAW_DET].std().median().to_numpy() + 1e-9
    scores_by_file = {}
    for f, g in dm.groupby("file"):
        X = g[RAW_DET].to_numpy()
        csum = np.vstack([np.zeros(len(RAW_DET)), np.cumsum(X, 0)])
        s = np.full(len(X), np.nan)
        for k in range(LAG_LO, len(X)):
            ref = (csum[k - LAG_HI] - csum[k - LAG_LO]) / (LAG_LO - LAG_HI)
            s[k] = np.mean(((X[k] - ref) / sd) ** 2)
        s = pd.Series(s).rolling(W_PER_S // 2, min_periods=1).median().to_numpy()  # 0.5 s median
        scores_by_file[f] = s
    f0 = [f for f in scores_by_file if f[1] == "0"][0]
    g0 = dm[dm.file == f0]
    thr = float(np.nanpercentile(scores_by_file[f0][(g0.t_center >= BASE_T + 1).to_numpy()], 99.5))
    for f, g in dm.groupby("file"):
        s = pd.Series(scores_by_file[f], index=g.index)
        t2_trace[f]["slide"] = np.nan_to_num(s.to_numpy(), nan=0.0).round(4).tolist()
        t2_trace[f]["slide_thr"] = thr
        valid = g.t_center >= BASE_T + 1
        onset = on.loc[f, "onset_t"]
        is_fault = g.fault.iloc[0] > 0
        pre_mask = valid & (g.t_center < onset - GUARD_T) if is_fault else valid
        r = {"file": f, "mode": mode, "fault": int(g.fault.iloc[0]), "threshold": thr,
             "false_alarm_rate_pre": float((s[pre_mask] > thr).mean()),
             "onset_t": float(onset) if is_fault else None, "detector": "sliding"}
        if is_fault:
            post = g.t_center >= onset
            alarm = s[post] > thr
            if alarm.any():
                r.update({"detected": True, "delay_s": float(g.t_center[post][alarm].iloc[0]) - onset,
                          "post_alarm_fraction": float(alarm.mean())})
            else:
                r.update({"detected": False, "delay_s": None, "post_alarm_fraction": 0.0})
        det2_rows.append(r)
det = pd.concat([det, pd.DataFrame(det2_rows)], ignore_index=True)
det.to_csv(RES / "detection_results.csv", index=False)

with open(RES / "predictions.json", "w") as fh:
    json.dump(preds, fh)
with open(RES / "t2_traces.json", "w") as fh:
    json.dump(t2_trace, fh)

# ---------- console summary ----------
agg = res.groupby(["exp", "scope", "features", "model"])[["acc", "bal_acc", "macro_f1"]].agg(["mean", "std"])
print(agg.round(3).to_string())
print(det.round(3).to_string())
print("E3 class counts:", e3.y_onset.value_counts().sort_index().to_dict())
