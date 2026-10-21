"""Cut every GPVS-Faults run into one-grid-cycle windows and compute features.

Window = 200 samples = 20 ms at the observed 100 us sampling period (one 50 Hz cycle).
Features per window:
  * "stat" set: mean, std, min, max of the 13 measured signals (52 features). This is the
    feature set used by Wali and Khan (2021), so we can reproduce their protocol.
  * "phys" set: a few physically meaningful quantities (DC power, AC-side apparent power proxy,
    phase-current unbalance, DC-link ripple, and the ratio of AC to DC power).

Run:  python code/02_features.py
Writes results/windows.csv  (one row per window)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "gpvs_csv" / "CSV_Files"
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)

SIGNALS = ["Ipv", "Vpv", "Vdc", "ia", "ib", "ic", "va", "vb", "vc", "Iabc", "If", "Vabc", "Vf"]
WIN = 200


def window_features(block: pd.DataFrame) -> dict:
    """Features for one 200-sample window."""
    out = {}
    for s in SIGNALS:
        x = block[s].to_numpy()
        out[f"{s}_mean"] = x.mean()
        out[f"{s}_std"] = x.std()
        out[f"{s}_min"] = x.min()
        out[f"{s}_max"] = x.max()
    # Physics-motivated features
    p_dc = (block["Vpv"] * block["Ipv"]).to_numpy()
    p_ac = (block["va"] * block["ia"] + block["vb"] * block["ib"] + block["vc"] * block["ic"]).to_numpy()
    i_rms = np.sqrt([np.mean(block[c].to_numpy() ** 2) for c in ("ia", "ib", "ic")])
    out["phys_Pdc"] = p_dc.mean()
    out["phys_Pac"] = p_ac.mean()
    out["phys_Pac_over_Pdc"] = p_ac.mean() / p_dc.mean() if abs(p_dc.mean()) > 1e-6 else 0.0
    out["phys_I_unbalance"] = (i_rms.max() - i_rms.min()) / i_rms.mean() if i_rms.mean() > 1e-6 else 0.0
    out["phys_Vdc_ripple"] = block["Vdc"].max() - block["Vdc"].min()
    out["phys_Irms_mean"] = i_rms.mean()
    return out


rows = []
for f in sorted(RAW.glob("F*.csv")):
    d = pd.read_csv(f)
    fault = int(f.stem[1])
    mode = "MPPT" if f.stem[2] == "M" else "IPPT"
    n_win = len(d) // WIN
    for k in range(n_win):
        block = d.iloc[k * WIN:(k + 1) * WIN]
        r = {"file": f.stem, "fault": fault, "mode": mode, "win": k,
             "t_center": block["Time"].iloc[WIN // 2]}
        r.update(window_features(block))
        rows.append(r)
    print(f.stem, n_win, "windows")

w = pd.DataFrame(rows)
w.to_csv(RES / "windows.csv", index=False)
print("total windows:", len(w), "features:", w.shape[1] - 5)
