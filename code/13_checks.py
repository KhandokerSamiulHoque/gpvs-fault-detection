"""Added robustness checks H1 to H4, plan in docs/checks_plan_2026-11-03.md.

  H1 label shuffle: training labels permuted, tested on true labels (RF on E2, LightGBM on E3b).
  H2 trivial baselines: DummyClassifier on the same E2 and E3b splits.
  H3 raw versus delta: paired block bootstrap on the stored seed-0 E3b predictions.
  H4 delta on the shortcut: RF on E2 windows from 3 s onward, raw statistics versus their delta versions.
  H6 all five E3b models refit over five seeds, logging default macro-F1 and macro-F1 over the six shared classes.

Run:  python code/13_checks.py [--smoke]
Writes one folder per run under results_checks/ and bootstrap_raw_vs_delta.json there.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_logger import RunLogger  # noqa: E402

SMOKE = "--smoke" in sys.argv
RES = ROOT / "results"
OUT = ROOT / ("results_checks_smoke" if SMOKE else "results_checks")
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
N_BOOT = 50 if SMOKE else 5000
BASE_T, GUARD_T, W_PER_S = 3.0, 0.5, 50
OUT.mkdir(exist_ok=True)

# ---------- same data, labels and features as 04_experiments.py ----------
w = pd.read_csv(RES / "windows.csv")
on = pd.read_csv(RES / "onsets.csv").set_index("file")
META = ["file", "fault", "mode", "win", "t_center"]
RAW_FEATS = [c for c in w.columns if c not in META]
STAT_FEATS = [c for c in RAW_FEATS if not c.startswith("phys_")]
w["onset_t"] = w["file"].map(on["onset_t"])
w["locatable"] = w["file"].map(on["locatable"])
w.loc[w.fault == 0, "onset_t"] = np.inf
w["post"] = w["t_center"] >= w["onset_t"]
w["near_onset"] = (w["t_center"] - w["onset_t"]).abs() < GUARD_T
w["y_onset"] = np.where(w["post"], w["fault"], 0)
base = w[w.t_center < BASE_T].groupby("file")[RAW_FEATS].mean()
DELTA_FEATS = [f"d_{c}" for c in RAW_FEATS]
w[DELTA_FEATS] = (w[RAW_FEATS] - base.loc[w["file"]].to_numpy()).to_numpy()
D_STAT = [f"d_{c}" for c in STAT_FEATS]


def rf(seed):
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)


def lgbm(seed):
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced",
                          random_state=seed, verbose=-1)


def models(seed):
    return {
        "RF": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed),
        "LightGBM": lgbm(seed),
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced",
                                                                     random_state=seed)),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced",
                                                       random_state=seed)),
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


def blocked(d):
    """First 70% of each run's windows train, last 30% test, as in E2."""
    tr, te = [], []
    for _, g in d.groupby("file"):
        cut = int(len(g) * 0.7)
        tr += list(g.index[:cut])
        te += list(g.index[cut:])
    return tr, te


def log(exp, method, scope, seed, model, Xtr, ytr, Xte, yte):
    # scope goes into a folder name, and Windows does not allow ">" there
    with RunLogger(exp=exp, method=method, dataset=scope.replace("->", "_to_"),
                   config={"model": type(model).__name__}, seed=seed, root=str(OUT)) as run:
        p = model.fit(Xtr, ytr).predict(Xte)
        f1 = f1_score(yte, p, average="macro")
        run.log_metrics({"macro_f1": f1, "n_test": len(yte), "n_classes": len(np.unique(yte))})
    print(f"{exp:18s} {scope:18s} {method:24s} seed {seed}  macro-F1 {f1:.3f}", flush=True)
    return f1


def permuted(y, seed):
    return np.random.default_rng(seed).permutation(y)


pre = w[(~w.post) & (~w.near_onset) & (w.locatable | (w.fault == 0))]
e3 = w[(w.t_center >= BASE_T) & (~w.near_onset) & (w.fault != 4)].copy()
DUMMIES = {"dummy_most_frequent": "most_frequent", "dummy_stratified": "stratified"}

# ---------- H1 and H2 on E2 ----------
for scope in ["MPPT", "IPPT"]:
    d = pre[pre["mode"] == scope]
    X, y = d[STAT_FEATS].to_numpy(), d["fault"].to_numpy()
    tr_b, te_b = blocked(d)
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        Xtr_b, ytr_b = d.loc[tr_b, STAT_FEATS].to_numpy(), d.loc[tr_b, "fault"].to_numpy()
        Xte_b, yte_b = d.loc[te_b, STAT_FEATS].to_numpy(), d.loc[te_b, "fault"].to_numpy()
        log("H1_E2_random", "rf_shuffled", scope, seed, rf(seed), Xtr, permuted(ytr, seed), Xte, yte)
        log("H1_E2_blocked", "rf_shuffled", scope, seed, rf(seed), Xtr_b, permuted(ytr_b, seed), Xte_b, yte_b)
        for name, strat in DUMMIES.items():
            log("H2_E2_random", name, scope, seed, DummyClassifier(strategy=strat, random_state=seed), Xtr, ytr, Xte, yte)
            log("H2_E2_blocked", name, scope, seed, DummyClassifier(strategy=strat, random_state=seed),
                Xtr_b, ytr_b, Xte_b, yte_b)

# ---------- H1 and H2 on E3b ----------
for tr_mode, te_mode in [("MPPT", "IPPT"), ("IPPT", "MPPT")]:
    scope = f"{tr_mode}->{te_mode}"
    tr, te = e3[e3["mode"] == tr_mode], e3[e3["mode"] == te_mode]
    ytr, yte = tr["y_onset"].to_numpy(), te["y_onset"].to_numpy()
    for seed in SEEDS:
        for fs, feats in [("raw", RAW_FEATS), ("delta", DELTA_FEATS)]:
            log("H1_E3b", f"lgbm_{fs}_shuffled", scope, seed, lgbm(seed), tr[feats].to_numpy(), permuted(ytr, seed),
                te[feats].to_numpy(), yte)
        for name, strat in DUMMIES.items():
            log("H2_E3b", name, scope, seed, DummyClassifier(strategy=strat, random_state=seed),
                tr[RAW_FEATS].to_numpy(), ytr, te[RAW_FEATS].to_numpy(), yte)

# ---------- H4: does delta remove the shortcut? ----------
late = pre[pre.t_center >= BASE_T]
for scope in ["MPPT", "IPPT"]:
    d = late[late["mode"] == scope]
    y = d["fault"].to_numpy()
    tr_b, te_b = blocked(d)
    for seed in SEEDS:
        for fs, feats in [("raw", STAT_FEATS), ("delta", D_STAT)]:
            log("H4_E2_blocked", f"rf_{fs}", scope, seed, rf(seed), d.loc[tr_b, feats].to_numpy(),
                d.loc[tr_b, "fault"].to_numpy(), d.loc[te_b, feats].to_numpy(), d.loc[te_b, "fault"].to_numpy())
            Xtr, Xte, ytr, yte = train_test_split(d[feats].to_numpy(), y, test_size=0.3, stratify=y, random_state=seed)
            log("H4_E2_random", f"rf_{fs}", scope, seed, rf(seed), Xtr, ytr, Xte, yte)
        for name, strat in DUMMIES.items():
            log("H4_E2_blocked", name, scope, seed, DummyClassifier(strategy=strat, random_state=seed),
                d.loc[tr_b, STAT_FEATS].to_numpy(), d.loc[tr_b, "fault"].to_numpy(),
                d.loc[te_b, STAT_FEATS].to_numpy(), d.loc[te_b, "fault"].to_numpy())

# ---------- H3: paired block bootstrap, raw versus delta ----------
preds = json.loads((RES / "predictions.json").read_text())
classes = sorted(e3.y_onset.unique())
SIX = [0, 1, 2, 5, 6, 7]   # classes with windows in both modes; F3 has none in MPPT


def macro_f1(y, p, labels=None):
    return f1_score(y, p, labels=labels or classes, average="macro", zero_division=0)


# ---------- H6: all five models on one scale, five seeds ----------
for tr_mode, te_mode in [("MPPT", "IPPT"), ("IPPT", "MPPT")]:
    tr, te = e3[e3["mode"] == tr_mode], e3[e3["mode"] == te_mode]
    yte = te["y_onset"].to_numpy()
    for fs, feats in [("raw", RAW_FEATS), ("delta", DELTA_FEATS)]:
        for seed in SEEDS:
            for mname, m in models(seed).items():
                with RunLogger(exp="H6_E3b", method=f"{mname}_{fs}", dataset=f"{tr_mode}_to_{te_mode}",
                               config={"model": mname, "features": fs}, seed=seed, root=str(OUT)) as run:
                    p = m.fit(tr[feats].to_numpy(), tr["y_onset"].to_numpy()).predict(te[feats].to_numpy())
                    d, s = f1_score(yte, p, average="macro"), macro_f1(yte, p, SIX)
                    run.log_metrics({"macro_f1": d, "macro_f1_six": s, "n_test": len(yte)})
                print(f"H6 {tr_mode}->{te_mode} {mname}_{fs} seed {seed} default {d:.3f} six {s:.3f}", flush=True)

rows = []
for tr_mode, te_mode in [("MPPT", "IPPT"), ("IPPT", "MPPT")]:
    te = e3[e3["mode"] == te_mode]
    files = te["file"].to_numpy()
    y_true = te["y_onset"].to_numpy()
    blocks = []
    for f in np.unique(files):
        idx = np.where(files == f)[0]
        blocks += [idx[i:i + W_PER_S] for i in range(0, len(idx), W_PER_S)]
    for mname in ["RF", "LightGBM", "LogReg", "SVM-RBF", "kNN"]:
        r = preds[f"E3b|raw|{mname}|{tr_mode}->{te_mode}"]
        s = preds[f"E3b|delta|{mname}|{tr_mode}->{te_mode}"]
        y, p_raw, p_delta = np.array(r["y"]), np.array(r["p"]), np.array(s["p"])
        assert np.array_equal(y, y_true) and np.array_equal(np.array(s["y"]), y_true), "stored predictions out of order"
        for score, labels in [("seven", classes), ("six", SIX)]:
            rng = np.random.default_rng(0)
            diffs = np.empty(N_BOOT)
            for b in range(N_BOOT):
                ii = np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))])
                diffs[b] = macro_f1(y[ii], p_delta[ii], labels) - macro_f1(y[ii], p_raw[ii], labels)
            p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
            f_raw, f_delta = macro_f1(y, p_raw, labels), macro_f1(y, p_delta, labels)
            rows.append({"score": score, "scope": f"{tr_mode}->{te_mode}", "model": mname, "f1_raw": f_raw,
                         "f1_delta": f_delta, "diff": f_delta - f_raw,
                         "ci_lo": float(np.percentile(diffs, 2.5)), "ci_hi": float(np.percentile(diffs, 97.5)),
                         "p": float(p), "n_boot": N_BOOT})
            print(rows[-1], flush=True)

# Holm over the 10 pairs, separately for each score
for score in ["seven", "six"]:
    fam = [r for r in rows if r["score"] == score]
    running = 0.0
    for rank, r in enumerate(sorted(fam, key=lambda r: r["p"])):
        running = max(running, min(1.0, (len(fam) - rank) * r["p"]))
        r["p_holm"] = running
(OUT / "bootstrap_raw_vs_delta.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

# ---------- summary ----------
meta = [json.loads(f.read_text(encoding="utf-8")) for f in OUT.glob("*/metrics.json")]
s = pd.DataFrame([{"exp": m["exp"], "scope": m["dataset"], "method": m["method"], "seed": m["seed"],
                   "macro_f1": m["metrics"]["macro_f1"]} for m in meta])
print(s.groupby(["exp", "scope", "method"])["macro_f1"].agg(["mean", "std", "count"]).round(3).to_string())
print(pd.DataFrame(rows).round(3).to_string())
