"""Thin orchestrator that runs train.py for every config in a directory
and collects their val_summary into one comparison table -- this is what
actually produces results/tables/modality_comparison.csv (Experiment 1,
spec section 14) and results/tables/model_comparison.csv (Experiment 2,
spec section 15). It does not reimplement training; it just calls
src/training/train.py per config and aggregates train_summary.json.

Usage:
    python src/experiments/run_experiment.py \
        --configs-dir configs/E02_modality \
        --exp-root experiments/E02_modality \
        --out results/tables/modality_comparison.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", required=True, help="Directory of *.yaml configs, one per condition.")
    ap.add_argument("--exp-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    configs_dir = Path(args.configs_dir)
    exp_root = Path(args.exp_root)
    rows = []

    for cfg_path in sorted(configs_dir.glob("*.yaml")):
        condition_name = cfg_path.stem
        exp_dir = exp_root / condition_name
        print(f"[run_experiment] === {condition_name} ===")
        result = subprocess.run(
            [sys.executable, "src/training/train.py", "--config", str(cfg_path), "--exp-dir", str(exp_dir)],
            check=False,
        )
        if result.returncode != 0:
            rows.append({"condition": condition_name, "status": "FAILED"})
            continue

        summary = json.loads((exp_dir / "train_summary.json").read_text())
        cfg = yaml.safe_load(open(cfg_path))
        rows.append({
            "condition": condition_name,
            "representation": cfg.get("representation"),
            "feature_set": cfg.get("feature_set"),
            "model": cfg.get("model"),
            "status": "OK",
            **summary,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[run_experiment] wrote comparison table -> {out_path}")
    print("[run_experiment] NOTE: this table reports VALIDATION metrics only, per spec "
          "section 13 (test set is not touched until the final model is selected). "
          "Add test-set numbers only after research/FINAL_CONFIGURATION.md is frozen "
          "and src/training/evaluate.py has been run once.")


if __name__ == "__main__":
    main()
