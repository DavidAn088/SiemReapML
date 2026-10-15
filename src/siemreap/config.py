"""Configuration, paths, run mode and deterministic seeds."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tomllib
from importlib import metadata
from pathlib import Path

MODES = ("full", "smoke")


def find_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: cwd) to the folder holding config/project.toml."""
    p = (start or Path.cwd()).resolve()
    for cand in (p, *p.parents):
        if (cand / "config" / "project.toml").exists():
            return cand
    raise FileNotFoundError("config/project.toml not found above " + str(p))


def load_config(root: Path | None = None) -> dict:
    root = root or find_root()
    cfg = tomllib.loads((root / "config" / "project.toml").read_text(encoding="utf-8"))
    cfg["_root"] = root
    return cfg


def run_mode() -> str:
    """'full' (default) or 'smoke', from the SIEMREAP_MODE environment variable."""
    mode = os.environ.get("SIEMREAP_MODE", "full").strip().lower()
    if mode not in MODES:
        raise ValueError(f"SIEMREAP_MODE must be one of {MODES}, got {mode!r}")
    return mode


def path(cfg: dict, key: str) -> Path:
    return cfg["_root"] / cfg["paths"][key]


def processed_dir(cfg: dict) -> Path:
    d = path(cfg, "processed")
    d.mkdir(parents=True, exist_ok=True)
    return d


def folds_dir(cfg: dict) -> Path:
    d = path(cfg, "folds")
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_dir(cfg: dict, stage: str, kind: str = "aggregate", mode: str | None = None) -> Path:
    """outputs/<stage>/<kind>, or outputs/smoke/<stage>/<kind> in smoke mode.

    kind = 'aggregate' (no borehole identifiers or coordinates; committable) or
    'restricted' (borehole-level; git-ignored) or '' for the stage folder itself."""
    mode = mode or run_mode()
    base = path(cfg, "outputs") / ("smoke" if mode == "smoke" else "") / stage
    d = base / kind if kind else base
    d.mkdir(parents=True, exist_ok=True)
    return d


def seed_for(cfg: dict, *parts) -> int:
    """Stable seed derived from the global seed and any identifying parts."""
    key = json.dumps([cfg["seeds"]["global"], *map(str, parts)])
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_record() -> dict:
    """Software versions, for the reproducibility record written next to results."""
    pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "optuna", "geopandas",
            "shapely", "rasterio", "pyproj", "matplotlib", "joblib", "xarray", "netCDF4"]
    vers = {}
    for p in pkgs:
        try:
            vers[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            vers[p] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": vers,
            "mode": run_mode()}


def smoke_banner(mode: str | None = None) -> str:
    mode = mode or run_mode()
    if mode == "smoke":
        return ("SMOKE TEST - software validation run with reduced budgets. "
                "Outputs are NOT experimental results and must not be reported.")
    return "FULL RUN"
