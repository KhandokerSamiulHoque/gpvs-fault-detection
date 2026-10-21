"""Estimate when the fault was introduced in each GPVS-Faults run.

The dataset README says faults were introduced "halfway" but gives no onset times, and the overview
plot shows onsets between about 7 s and 11.5 s. We locate a single change point per run with a
multivariate least-squares split (the statistic used in binary segmentation):

    S(k) = sum_j  n1*n2/n * (mean_before_j - mean_after_j)^2      over z-scored features j

searched over windows from 3 s to 0.5 s before the end. The fault-free runs (F0L, F0M) give the size
of S for ordinary drift with no fault, so a run whose normalized change score is not clearly above
the F0 level has no locatable onset.

Run:  python code/03_onset.py
Writes results/onsets.csv and figures/onsets_vdc.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

CP_FEATURES = ["Ipv_mean", "Vpv_mean", "Vdc_mean", "Iabc_mean", "Ipv_std", "Vpv_std",
               "Vdc_std", "ia_std", "ib_std", "ic_std", "phys_I_unbalance"]
W_PER_S = 50  # 20 ms windows

w = pd.read_csv(RES / "windows.csv")
rows = []
for name, g in w.groupby("file", sort=True):
    X = g[CP_FEATURES].to_numpy()
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    n = len(X)
    lo, hi = 3 * W_PER_S, n - W_PER_S // 2
    csum = np.cumsum(X, axis=0)
    total = csum[-1]
    best_k, best_s = None, -np.inf
    for k in range(lo, hi):
        n1, n2 = k, n - k
        m1 = csum[k - 1] / n1
        m2 = (total - csum[k - 1]) / n2
        s = n1 * n2 / n * np.sum((m1 - m2) ** 2)
        if s > best_s:
            best_k, best_s = k, s
    # Step size per feature: |median of next 2 s - median of previous 2 s|, in units of that feature's
    # standard deviation over the first 3 s of the run. 2 s is longer than the ~1.4 s oscillation seen in
    # every run, so the oscillation averages out. A fault that moves only one signal (e.g. a Vdc step)
    # is diluted in the combined score above; this per-feature step catches it.
    raw = g[CP_FEATURES].to_numpy()
    base_sd = raw[:lo].std(0) + 1e-12
    half = 2 * W_PER_S
    steps = np.zeros(len(CP_FEATURES))
    for k in range(max(lo, half), min(hi, n - half)):
        d = np.abs(np.median(raw[k:k + half], 0) - np.median(raw[k - half:k], 0)) / base_sd
        steps = np.maximum(steps, d)
    rows.append({"file": name, "fault": int(name[1]), "onset_win": best_k,
                 "onset_t": g["t_center"].iloc[best_k], "score": best_s / n,
                 "step_max_sd": steps.max(),
                 "top_feature": CP_FEATURES[int(steps.argmax())], "n_win": n})

on = pd.DataFrame(rows)
f0_max = on.loc[on.fault == 0, "score"].max()
f0_step_max = on.loc[on.fault == 0, "step_max_sd"].max()
on["score_vs_F0"] = on["score"] / f0_max
on["feat_score_vs_F0"] = on["step_max_sd"] / f0_step_max
# A run counts as having a locatable onset if either its combined or its strongest single-feature
# change score is at least 2x the largest value seen in the fault-free runs.
on["locatable"] = (on["fault"] > 0) & ((on["score_vs_F0"] >= 2.0) | (on["feat_score_vs_F0"] >= 2.0))
on.to_csv(RES / "onsets.csv", index=False)
print(on.round(3).to_string())

# Figure: Vdc per run with the estimated onset marked. Sized to print at 16 cm wide (TJEECS limit),
# Times New Roman, saved as PNG (drafts) and PDF (TJEECS accepts only PDF figures).
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7, "pdf.fonttype": 42})  # 42 = editable TrueType text
fig, axes = plt.subplots(4, 4, figsize=(6.3, 5.6))
for ax, (_, r) in zip(axes.ravel(), on.iterrows()):
    g = w[w.file == r.file]
    ax.plot(g.t_center, g.Vdc_mean, lw=0.6, color="#1f4e79")
    if r.fault > 0:
        ax.axvline(r.onset_t, color="#c0392b" if r.locatable else "#999999", ls="--", lw=0.9)
    ax.set_title(r.file, fontsize=7, pad=2)
    ax.tick_params(labelsize=6, length=2, pad=1)
for ax in axes[-1]:
    ax.set_xlabel("Time (s)", fontsize=7)
for ax in axes[:, 0]:
    ax.set_ylabel("Vdc (V)", fontsize=7)
fig.tight_layout(h_pad=0.6, w_pad=0.4)
fig.savefig(FIG / "onsets_vdc.png", dpi=300)
fig.savefig(FIG / "onsets_vdc.pdf")
