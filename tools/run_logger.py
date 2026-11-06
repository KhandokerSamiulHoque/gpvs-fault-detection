"""Reproducible run logging: every run gets results/<run_id>/ with config, env, git state, metrics, predictions.

Example:
    from tools.run_logger import RunLogger
    with RunLogger(exp="pv_forecast", method="lightgbm", config=cfg, seed=0, fold=0, dataset="siteA") as run:
        ...
        run.log_metrics({"mae": 12.3, "rmse": 18.9})
        run.save_predictions(y_true, y_pred, ids=ids)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


class RunLogger:
    def __init__(self, exp: str, method: str, config: dict[str, Any], seed: int,
                 fold: int | None = None, dataset: str = "default", root: str = "results",
                 data_files: list[str] | None = None):
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{exp}__{dataset}__{method}__s{seed}" + (f"_f{fold}" if fold is not None else "") + f"__{stamp}"
        self.dir = Path(root) / self.run_id
        self.meta = {"run_id": self.run_id, "exp": exp, "method": method, "dataset": dataset,
                     "seed": seed, "fold": fold, "config": config, "metrics": {}}
        self.data_files = data_files or []

    def __enter__(self) -> "RunLogger":
        self.dir.mkdir(parents=True, exist_ok=False)
        self._t0 = time.perf_counter()
        self.meta["git_commit"] = _git("rev-parse", "HEAD")
        self.meta["git_dirty"] = _git("status", "--porcelain") not in ("", "unavailable")
        self.meta["data_sha256"] = {p: file_sha256(p) for p in self.data_files}
        env = [f"python: {sys.version}", f"platform: {platform.platform()}", f"processor: {platform.processor()}"]
        try:
            import torch  # type: ignore
            env.append(f"torch: {torch.__version__}, cuda: {torch.version.cuda}, "
                       f"gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
        except ImportError:
            pass
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        (self.dir / "env.txt").write_text("\n".join(env) + "\n\n" + freeze)
        try:
            import yaml  # type: ignore
            (self.dir / "config.yaml").write_text(yaml.safe_dump(self.meta["config"], sort_keys=False))
        except ImportError:
            (self.dir / "config.json").write_text(json.dumps(self.meta["config"], indent=2, default=str))
        return self

    def log_metrics(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            v = float(v)
            if v != v:  # NaN
                raise ValueError(f"metric {k} is NaN — investigate, do not report")
            self.meta["metrics"][k] = v

    def save_predictions(self, y_true, y_pred, ids=None) -> None:
        import numpy as np
        np.savez_compressed(self.dir / "predictions.npz", y_true=np.asarray(y_true),
                            y_pred=np.asarray(y_pred), ids=np.asarray(ids) if ids is not None else np.array([]))

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.meta["wall_clock_s"] = round(time.perf_counter() - self._t0, 3)
        self.meta["status"] = "failed" if exc_type else "ok"
        if exc_type:
            self.meta["error"] = f"{exc_type.__name__}: {exc}"
        (self.dir / "metrics.json").write_text(json.dumps(self.meta, indent=2, default=str))
        return False  # never swallow exceptions
