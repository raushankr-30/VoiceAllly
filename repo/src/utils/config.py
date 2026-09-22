"""Load / save experiment configs (YAML) and stamp every run with the
config actually used, per spec section 26 (reproducibility).

Every training / evaluation script loads its config through `load_config`
and writes it back into the experiment's own directory via
`save_config_snapshot` so that experiments/E0X_.../config.yaml is always
an exact record of what produced that experiment's results -- not the
template it started from.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_source_config_path"] = str(path)
    return cfg


def save_config_snapshot(cfg: Dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = copy.deepcopy(cfg)
    dest = out_dir / "config.yaml"
    with open(dest, "w") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False)
    return dest


def merge_overrides(base_cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-merge CLI/programmatic overrides on top of a base config."""
    merged = copy.deepcopy(base_cfg)
    merged.update(overrides)
    return merged
