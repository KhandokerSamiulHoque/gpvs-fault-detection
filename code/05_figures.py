"""Publication figures for Paper B (300 dpi PNG). Reads the outputs of 03_onset.py and 04_experiments.py.

Run:  python code/05_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

# Palette (validated reference palette, light mode): categorical slots 1-3, ink, grid, sequential blue.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": False, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})
CLASS_LABEL = {0: "Normal", 1: "F1 inverter", 2: "F2 sensor", 3: "F3 grid", 5: "F5 open-circ.",
               6: "F6 PI gain", 7: "F7 PI time-const."}



def save(fig, name):
    """Save PNG (for the Word/Markdown drafts) and PDF (TJEECS accepts only PDF figures)."""
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")


res = pd.read_csv(RES / "experiment_results.csv")
preds = json.load(open(RES / "predictions.json"))

# ---------- Fig. 3: the same data, four evaluation protocols ----------
def ms(q):
    s = res.query(q)["macro_f1"]
    return s.mean(), s.std()

bars = [
    ("E1\nrun labels\nrandom split", ms("exp=='E1_file_label_random' and scope=='MPPT'"), None),
    ("E2\nfault-free only\nrandom split", ms("exp=='E2_prefault_random' and scope=='MPPT'"), 1 / 7),
    ("E2\nfault-free only\ntime-blocked", ms("exp=='E2_prefault_blocked' and scope=='MPPT'"), 1 / 7),
    ("E3a\nonset labels\nrandom split", ms("exp=='E3a_onset_random' and model=='LightGBM' and features=='raw'"), None),
    ("E3b\nMPPT→IPPT\nraw features", ms("exp=='E3b_cross_mode' and model=='LightGBM' and features=='raw' and scope=='MPPT->IPPT'"), None),
    ("E3b\nMPPT→IPPT\ndelta features", ms("exp=='E3b_cross_mode' and model=='LightGBM' and features=='delta' and scope=='MPPT->IPPT'"), None),
]
fig, ax = plt.subplots(figsize=(7.2, 3.2))
x = np.arange(len(bars))
cols = [ORANGE, ORANGE, ORANGE, BLUE, BLUE, BLUE]
for i, (lab, (m, s), chance) in enumerate(bars):
    ax.bar(i, m, width=0.62, color=cols[i], yerr=s if s > 0 else None, ecolor=INK2, capsize=3, zorder=2)
    ax.text(i, m + 0.03, f"{m:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    if chance:
        ax.plot([i - 0.31, i + 0.31], [chance, chance], color=INK, lw=1.2, ls=":", zorder=3)
ax.text(1.5, 1 / 7 + 0.03, "chance (7 runs)", ha="center", fontsize=7.5, color=INK2)
ax.set_xticks(x, [b[0] for b in bars], fontsize=7.5)
ax.set_ylabel("Macro-F1")
ax.set_ylim(0, 1.12)
ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
          labels=["Label = run identity (shortcut possible)", "Label = physical state (onset-based)"],
          fontsize=7.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.13), ncol=2)
save(fig, "fig3_protocols")
plt.close(fig)

# ---------- Fig. 4: cross-mode macro-F1 heatmap, models x (feature set, direction) ----------
e3b = res[res.exp == "E3b_cross_mode"].groupby(["model", "features", "scope"])["macro_f1"].mean().unstack(["features", "scope"])
order_cols = [("raw", "MPPT->IPPT"), ("raw", "IPPT->MPPT"), ("delta", "MPPT->IPPT"), ("delta", "IPPT->MPPT")]
order_rows = ["LightGBM", "RF", "kNN", "SVM-RBF", "LogReg"]
H = e3b.loc[order_rows, order_cols]
fig, ax = plt.subplots(figsize=(5.2, 2.8))
im = ax.imshow(H.to_numpy(), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = H.iat[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="white" if v > 0.6 else INK)
ax.set_xticks(range(4), [f"{f}\n{s.replace('->', chr(8594))}" for f, s in order_cols], fontsize=8)
ax.set_yticks(range(len(order_rows)), order_rows)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Macro-F1 (mean of 5 seeds)", fontsize=8)
cb.outline.set_visible(False)
for s in ax.spines.values():
    s.set_visible(False)
save(fig, "fig4_crossmode_heatmap")
plt.close(fig)

# ---------- Fig. 5: confusion matrices, random split vs cross-mode ----------
def cm_panel(ax, key, title):
    y, p = np.array(preds[key]["y"]), np.array(preds[key]["p"])
    labs = sorted(CLASS_LABEL)
    cm = confusion_matrix(y, p, labels=labs).astype(float)
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(cm, cmap=SEQ, vmin=0, vmax=1)
    for i in range(len(labs)):
        for j in range(len(labs)):
            if cm[i, j] >= 0.005:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > 0.6 else INK)
    names = [CLASS_LABEL[c] for c in labs]
    ax.set_xticks(range(len(labs)), names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labs)), names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_title(title, fontsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
cm_panel(axes[0], "E3a|delta|LightGBM", "(a) Random window split")
cm_panel(axes[1], "E3b|delta|LightGBM|MPPT->IPPT", "(b) Train MPPT, test IPPT")
axes[0].set_ylabel("True (row-normalized)")
fig.tight_layout()
save(fig, "fig5_confusion")
plt.close(fig)

# ---------- Fig. 6: Hotelling T^2 traces ----------
tr = json.load(open(RES / "t2_traces.json"))
det = pd.read_csv(RES / "detection_results.csv")
show = ["F0M", "F2M", "F7M", "F6M", "F4L", "F2L"]
fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.9), sharex=True)
for ax, f in zip(axes.ravel(), show):
    t = np.array(tr[f]["t"])
    keep_t = t >= 0.5  # drop the start-up transient in the first cycles
    static = np.array(tr[f]["t2"]) / tr[f]["thr"]
    slide = np.array(tr[f]["slide"]) / tr[f]["slide_thr"]
    ax.plot(t[keep_t], np.clip(static[keep_t], 1e-2, None), color=BLUE, lw=1.0)
    m4 = t >= 4.0  # sliding detector needs 4 s of history
    ax.plot(t[m4], np.clip(slide[m4], 1e-2, None), color=AQUA, lw=1.0)
    ax.axhline(1.0, color=INK, ls=":", lw=1.0)
    if f[1] != "0":
        ax.axvline(det.query("file==@f and detector=='static'")["onset_t"].iloc[0], color=ORANGE, lw=1.2, ls="--")
    ax.set_yscale("log")
    ax.set_title(f, fontsize=8.5)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
for ax in axes[1]:
    ax.set_xlabel("Time (s)")
for ax in axes[:, 0]:
    ax.set_ylabel("Score / threshold")
fig.legend(handles=[plt.Line2D([], [], color=BLUE), plt.Line2D([], [], color=AQUA),
                    plt.Line2D([], [], color=INK, ls=":"), plt.Line2D([], [], color=ORANGE, ls="--")],
           labels=["Static baseline T²", "Sliding baseline (t−4 s to t−1 s)",
                   "Alarm threshold", "Estimated onset"],
           loc="upper center", ncol=4, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout()
save(fig, "fig6_t2_detection")
plt.close(fig)

# ---------- Fig. 7: SHAP, delta LightGBM trained on MPPT ----------
w = pd.read_csv(RES / "windows.csv")
on = pd.read_csv(RES / "onsets.csv").set_index("file")
META = ["file", "fault", "mode", "win", "t_center"]
RAW_FEATS = [c for c in w.columns if c not in META]
base = w[w.t_center < 3.0].groupby("file")[RAW_FEATS].mean()
D = w[RAW_FEATS] - base.loc[w["file"]].to_numpy()
w["onset_t"] = w["file"].map(on["onset_t"])
w.loc[w.fault == 0, "onset_t"] = np.inf
w["y"] = np.where(w.t_center >= w.onset_t, w.fault, 0)
keep = (w.t_center >= 3.0) & ((w.t_center - w.onset_t).abs() >= 0.5) & (w.fault != 4)
tr_mask = keep & (w["mode"] == "MPPT")
m = LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced", random_state=0, verbose=-1)
m.fit(D[tr_mask], w.loc[tr_mask, "y"])
# TreeSHAP values from LightGBM itself (same algorithm as shap.TreeExplainer). Output is
# n x (classes * (features + 1)); the last column of each class block is the bias term.
contrib = m.predict(D[tr_mask], pred_contrib=True)
nf = len(RAW_FEATS)
sv = np.abs(np.asarray(contrib).reshape(len(contrib), -1, nf + 1)[:, :, :nf])  # (n, classes, features)
imp = sv.mean(axis=(0, 1))
imp = pd.Series(imp, index=RAW_FEATS).sort_values(ascending=True).tail(15)
imp.to_csv(RES / "shap_top15_delta_lgbm_mppt.csv")
fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.barh(range(len(imp)), imp.values, color=BLUE, height=0.62)
ax.set_yticks(range(len(imp)), ["Δ " + c for c in imp.index], fontsize=7.5)
ax.set_xlabel("Mean |SHAP value| (all classes)")
ax.xaxis.grid(True, color=GRID, lw=0.5)
save(fig, "fig7_shap")
plt.close(fig)
print("figures written; top SHAP features:", list(imp.index[::-1][:5]))
