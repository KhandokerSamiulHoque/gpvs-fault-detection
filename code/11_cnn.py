"""1D CNN on raw 20 ms waveforms for E2 (shortcut probe) and E3b (cross-mode).

Input: each window is 13 signals x 200 samples, taken straight from the CSV files (no hand-made features).
Labels, splits, guard band (0.5 s) and baseline (first 3 s) are the same as in 04_experiments.py.
  E2  fault-free windows before onset, label = the run's fault id; random 70/30 and time-blocked 70/30, per mode.
  E3b onset-based labels, F4 excluded, t >= 3 s; train MPPT -> test IPPT and the reverse;
      "raw" input and "delta" input (each channel minus its mean over the run's first 3 s).
Channels are standardized with training-set statistics only. Three seeds per setting.

Run:  python code/11_cnn.py
Writes results/cnn_results.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "gpvs_csv" / "CSV_Files"
RES = ROOT / "results"
SIGNALS = ["Ipv", "Vpv", "Vdc", "ia", "ib", "ic", "va", "vb", "vc", "Iabc", "If", "Vabc", "Vf"]
WIN, BASE_T, GUARD_T = 200, 3.0, 0.5
SEEDS = [int(s) for s in sys.argv[1:]] or [0, 1, 2]   # seeds on the command line go to cnn_results_seeds.csv
SIX = [0, 1, 2, 5, 6, 7]   # classes with windows in both modes; F3 has none in MPPT
EPOCHS, BATCH, LR = 15, 128, 1e-3

# ---------- load all windows as arrays ----------
on = pd.read_csv(RES / "onsets.csv").set_index("file")
X_list, meta = [], []
for f in sorted(RAW.glob("F*.csv")):
    d = pd.read_csv(f)
    n = len(d) // WIN
    arr = d[SIGNALS].to_numpy(np.float32)[: n * WIN].reshape(n, WIN, len(SIGNALS)).transpose(0, 2, 1)
    t = d["Time"].to_numpy()[: n * WIN].reshape(n, WIN)[:, WIN // 2]
    X_list.append(arr)
    meta.append(pd.DataFrame({"file": f.stem, "fault": int(f.stem[1]),
                              "mode": "MPPT" if f.stem[2] == "M" else "IPPT", "t_center": t}))
X = np.concatenate(X_list)                      # (n_windows, 13, 200)
m = pd.concat(meta, ignore_index=True)
onset = m["file"].map(on["onset_t"]).to_numpy(dtype=float, copy=True)
onset[m.fault.to_numpy() == 0] = np.inf
locatable = m["file"].map(on["locatable"]).to_numpy().astype(bool)
post = m.t_center.to_numpy() >= onset
near = np.abs(m.t_center.to_numpy() - onset) < GUARD_T
y_onset = np.where(post, m.fault.to_numpy(), 0)

# delta input: subtract each run's per-channel mean over its first 3 s
X_delta = X.copy()
for f in m.file.unique():
    idx = (m.file == f).to_numpy()
    base = X[idx & (m.t_center.to_numpy() < BASE_T)].mean(axis=(0, 2), keepdims=True)
    X_delta[idx] -= base
print("windows:", X.shape)


class CNN(nn.Module):
    def __init__(self, n_in, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_in, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.net(x)


def fit_predict(Xtr, ytr, Xte, seed):
    """Standardize channels on the training set, train with class-weighted loss, predict test labels."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sd)
    Xte_t = torch.tensor((Xte - mu) / sd)
    classes = np.unique(ytr)
    ytr_i = torch.tensor(np.searchsorted(classes, ytr))
    counts = np.bincount(ytr_i.numpy(), minlength=len(classes))
    w = torch.tensor(len(ytr) / (len(classes) * counts), dtype=torch.float32)
    model = CNN(Xtr.shape[1], len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(Xtr_t[b]), ytr_i[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.cat([model(Xte_t[i:i + 1024]).argmax(1) for i in range(0, len(Xte_t), 1024)])
    return classes[p.numpy()]


def scores(y, p):
    return {"acc": float(np.mean(y == p)), "bal_acc": float(balanced_accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro")),
            "macro_f1_six": float(f1_score(y, p, labels=SIX, average="macro", zero_division=0))}


rows = []
# ---------- E2: shortcut probe on fault-free windows ----------
pre = (~post) & (~near) & (locatable | (m.fault.to_numpy() == 0))
for mode in ["MPPT", "IPPT"]:
    sel = np.where(pre & (m["mode"] == mode).to_numpy())[0]
    y = m.fault.to_numpy()[sel]
    for seed in SEEDS:
        tr, te = train_test_split(sel, test_size=0.3, stratify=y, random_state=seed)
        p = fit_predict(X[tr], m.fault.to_numpy()[tr], X[te], seed)
        rows.append({"exp": "E2_prefault_random", "scope": mode, "input": "raw", "seed": seed,
                     "n_classes": len(np.unique(y)), **scores(m.fault.to_numpy()[te], p)})
        tr_b, te_b = [], []
        for f in pd.unique(m.file.to_numpy()[sel]):
            idx = sel[m.file.to_numpy()[sel] == f]
            cut = int(len(idx) * 0.7)
            tr_b += list(idx[:cut]); te_b += list(idx[cut:])
        p = fit_predict(X[tr_b], m.fault.to_numpy()[tr_b], X[te_b], seed)
        rows.append({"exp": "E2_prefault_blocked", "scope": mode, "input": "raw", "seed": seed,
                     "n_classes": len(np.unique(y)), **scores(m.fault.to_numpy()[te_b], p)})
        print("E2", mode, seed, rows[-2]["macro_f1"], rows[-1]["macro_f1"])

# ---------- E3b: cross-mode with onset labels ----------
keep = (m.t_center.to_numpy() >= BASE_T) & (~near) & (m.fault.to_numpy() != 4)
for name, XX in [("raw", X), ("delta", X_delta)]:
    for tr_mode, te_mode in [("MPPT", "IPPT"), ("IPPT", "MPPT")]:
        tr = np.where(keep & (m["mode"] == tr_mode).to_numpy())[0]
        te = np.where(keep & (m["mode"] == te_mode).to_numpy())[0]
        for seed in SEEDS:
            p = fit_predict(XX[tr], y_onset[tr], XX[te], seed)
            rows.append({"exp": "E3b_cross_mode", "scope": f"{tr_mode}->{te_mode}", "input": name,
                         "seed": seed, **scores(y_onset[te], p)})
            print("E3b", name, tr_mode, seed, rows[-1]["macro_f1"])

res = pd.DataFrame(rows)
res.to_csv(RES / ("cnn_results_seeds.csv" if sys.argv[1:] else "cnn_results.csv"), index=False)
print(res.groupby(["exp", "scope", "input"])["macro_f1"].agg(["mean", "std"]).round(3).to_string())
