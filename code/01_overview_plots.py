"""Overview of every GPVS-Faults experiment and two data-quality checks.

1. Plots 50 ms moving means of Ipv, Vpv, Vdc and Iabc for all 16 files, so the fault onset is visible.
2. Checks whether the phase voltage columns va and vb carry distinct signals.

Run:  python code/01_overview_plots.py
Writes figures/overview_all_files.png and results/voltage_channel_check.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "gpvs_csv" / "CSV_Files"
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

SIGNALS = ["Ipv", "Vpv", "Vdc", "Iabc"]
WIN = 500  # samples, 50 ms at the observed 10 kHz rate

files = sorted(RAW.glob("F*.csv"))
fig, axes = plt.subplots(len(files), len(SIGNALS), figsize=(14, 2.0 * len(files)), sharex="row")
checks = []
for i, f in enumerate(files):
    d = pd.read_csv(f)
    for j, s in enumerate(SIGNALS):
        ax = axes[i, j]
        ax.plot(d["Time"], d[s].rolling(WIN, center=True).mean(), lw=0.8, color="#1f4e79")
        if i == 0:
            ax.set_title(s)
        if j == 0:
            ax.set_ylabel(f.stem, rotation=0, labelpad=22, fontsize=10)
        ax.tick_params(labelsize=7)
    # Data-quality check: are va and vb the same signal?
    checks.append({
        "file": f.stem,
        "corr_va_vb": np.corrcoef(d["va"], d["vb"])[0, 1],
        "corr_va_vc": np.corrcoef(d["va"], d["vc"])[0, 1],
        "max_abs_va_minus_vb": (d["va"] - d["vb"]).abs().max(),
        "corr_ia_ib": np.corrcoef(d["ia"], d["ib"])[0, 1],
    })
for ax in axes[-1]:
    ax.set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig(FIG / "overview_all_files.png", dpi=110)

pd.DataFrame(checks).to_csv(RES / "voltage_channel_check.csv", index=False)
print(pd.DataFrame(checks).round(4).to_string())
