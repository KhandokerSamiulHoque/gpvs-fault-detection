"""Summarize the added checks: tables for H1, H2, H4, H6 and the H3 bootstrap from 13_checks.py, H5 from the CNN seeds.

Run:  python code/14_checks_summary.py
Writes results_checks/summary_h5.json and prints every table. Plan: notes/prereg_checks_2026-11-03.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "results_checks"
pd.set_option("display.width", 200)


def holm(ps):
    order, adj, running = np.argsort(ps), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


meta = [json.loads(f.read_text(encoding="utf-8")) for f in OUT.glob("*/metrics.json")]
assert all(m["status"] == "ok" for m in meta)
s = pd.DataFrame([{"exp": m["exp"], "scope": m["dataset"], "method": m["method"], "seed": m["seed"], **m["metrics"]}
                  for m in meta])
print("runs:", len(s))
print(s[s.exp != "H6_E3b"].groupby(["exp", "scope", "method"])["macro_f1"]
      .agg(["mean", "std", "min", "max", "count"]).round(3).to_string())

h6 = s[s.exp == "H6_E3b"]
print("\nH6: all models, five seeds, both scores")
print(h6.groupby(["scope", "method"])[["macro_f1", "macro_f1_six"]].agg(["mean", "std"]).round(3).to_string())
pub = pd.read_csv(RES / "experiment_results.csv")
pub = pub[pub.exp == "E3b_cross_mode"]
pub_m = pub.assign(k=pub.scope.str.replace("->", "_to_") + "|" + pub.model + "_" + pub.features)
h6_m = h6.assign(k=h6.scope + "|" + h6.method)
merged = pub_m.merge(h6_m, on=["k", "seed"], suffixes=("_pub", "_h6"))
print("H6 reproduces published default macro-F1, largest absolute difference:",
      (merged.macro_f1_pub - merged.macro_f1_h6).abs().max(), "over", len(merged), "runs")

boot = pd.DataFrame(json.loads((OUT / "bootstrap_raw_vs_delta.json").read_text(encoding="utf-8")))
print("\nH3: paired block bootstrap, delta minus raw")
print(boot.round(4).to_string())

# 95% CI of the six-class score, same 1 s blocks as H3, seed-0 stored predictions
w = pd.read_csv(RES / "windows.csv", usecols=["file", "fault", "mode", "t_center"])
on = pd.read_csv(RES / "onsets.csv").set_index("file")
onset = w["file"].map(on["onset_t"]).to_numpy(dtype=float, copy=True)
onset[w.fault.to_numpy() == 0] = np.inf
e3 = w[(w.t_center >= 3.0) & (np.abs(w.t_center - onset) >= 0.5) & (w.fault != 4)]
preds = json.loads((RES / "predictions.json").read_text())
SIX = [0, 1, 2, 5, 6, 7]
cis = []
for tr_m, te_m in [("MPPT", "IPPT"), ("IPPT", "MPPT")]:
    files = e3[e3["mode"] == te_m]["file"].to_numpy()
    blocks = []
    for f in np.unique(files):
        idx = np.where(files == f)[0]
        blocks += [idx[i:i + 50] for i in range(0, len(idx), 50)]
    for fs in ["raw", "delta"]:
        v = preds[f"E3b|{fs}|LightGBM|{tr_m}->{te_m}"]
        y, p = np.array(v["y"]), np.array(v["p"])
        assert len(y) == len(files)
        rng = np.random.default_rng(0)
        vals = [f1_score(y[ii], p[ii], labels=SIX, average="macro", zero_division=0)
                for ii in (np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))])
                           for _ in range(5000))]
        cis.append({"scope": f"{tr_m}->{te_m}", "features": fs,
                    "f1": f1_score(y, p, labels=SIX, average="macro", zero_division=0),
                    "ci_lo": float(np.percentile(vals, 2.5)), "ci_hi": float(np.percentile(vals, 97.5))})
(OUT / "crossmode_six_ci.json").write_text(json.dumps(cis, indent=1), encoding="utf-8")
print("\nLightGBM six-class 95% CI, seed 0")
print(pd.DataFrame(cis).round(3).to_string())

cnn = pd.read_csv(RES / "cnn_results_seeds.csv")
print("\nH5: CNN, 10 seeds")
print(cnn.groupby(["exp", "scope", "input"])[["macro_f1", "macro_f1_six"]].agg(["mean", "std", "min", "max"])
      .round(3).to_string())
old = pd.read_csv(RES / "cnn_results.csv")
m = old.merge(cnn, on=["exp", "scope", "input", "seed"], suffixes=("_old", "_new"))
print("seeds 0-2 reproduce stored values, largest absolute difference:",
      (m.macro_f1_old - m.macro_f1_new).abs().max(), "over", len(m), "runs")

e3 = cnn[cnn.exp == "E3b_cross_mode"]
out = {"a_delta_vs_raw": [], "b_delta_vs_lightgbm": []}
dirs = ["MPPT->IPPT", "IPPT->MPPT"]
for score in ["macro_f1_six", "macro_f1"]:
    ps = []
    for d in dirs:
        g = e3[e3.scope == d].pivot(index="seed", columns="input", values=score)
        ps.append(float(stats.wilcoxon(g["delta"], g["raw"]).pvalue))
    for d, p, q in zip(dirs, ps, holm(ps)):
        g = e3[e3.scope == d].pivot(index="seed", columns="input", values=score)
        diff = g["delta"] - g["raw"]
        out["a_delta_vs_raw"].append({"score": score, "scope": d, "mean_diff": float(diff.mean()),
                                      "delta_wins": int((diff > 0).sum()), "n": len(diff), "p": p, "p_holm": q})
lgbm_six = h6[h6.method == "LightGBM_delta"].groupby("scope")["macro_f1_six"].mean()
ps, rows = [], []
for d in dirs:
    x = e3[(e3.scope == d) & (e3.input == "delta")]["macro_f1_six"].to_numpy()
    ref = float(lgbm_six[d.replace("->", "_to_")])
    ps.append(float(stats.wilcoxon(x - ref).pvalue))
    rows.append({"scope": d, "cnn_mean": float(x.mean()), "cnn_sd": float(x.std(ddof=1)), "cnn_min": float(x.min()),
                 "cnn_max": float(x.max()), "lightgbm": ref, "cnn_above": int((x > ref).sum()), "n": len(x)})
for r, p, q in zip(rows, ps, holm(ps)):
    r.update({"p": p, "p_holm": q})
out["b_delta_vs_lightgbm"] = rows
(OUT / "summary_h5.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
