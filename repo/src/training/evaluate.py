"""STEP 19: Untouched-test-set evaluation of ONE frozen model (spec
section 18). This script must only ever be pointed at
research/FINAL_CONFIGURATION.md's exact config, and only ever run once
per final model -- re-running it repeatedly against the test set to
"see how we're doing" defeats the entire point of the split protocol.

Produces:
  - results/tables/final_test_results.csv (or robustness.csv when
    --perturbation is set)
  - results/figures/confusion_matrix.png
  - results/figures/per_class_f1.png

Usage:
    python src/training/evaluate.py --config experiments/E07_final/config.yaml \
        --checkpoint experiments/E07_final/checkpoint_best.pt \
        --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.features.dataset import LandmarkSequenceDataset, expected_feature_dim
from src.models.architectures import build_model
from src.utils.config import load_config
from src.utils.metrics import (
    classification_metrics,
    confidence_interval_binomial,
    confusion,
    per_class_f1,
)
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["validation", "test"], default="test")
    ap.add_argument("--tables-out", default="results/tables")
    ap.add_argument("--figures-out", default="results/figures")
    ap.add_argument("--tag", default="final",
                     help="Filename tag, e.g. 'final', 'R1', 'robustness_lighting'.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = Path(args.checkpoint).parent
    label_to_idx = json.loads((exp_dir / "label_to_idx.json").read_text())
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    split_csv = cfg["val_csv"] if args.split == "validation" else cfg["test_csv"]
    ds = LandmarkSequenceDataset(
        split_csv=split_csv, landmarks_dir=cfg["landmarks_dir"], video_col=cfg["video_col"],
        label_col=cfg["label_col"], label_to_idx=label_to_idx, representation=cfg["representation"],
        feature_set=cfg["feature_set"], sequence_length=cfg["sequence_length"],
    )
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False)

    input_dim = expected_feature_dim(cfg["representation"], cfg["feature_set"])
    model = build_model(cfg["model"], input_dim=input_dim, num_classes=len(class_names),
                         **cfg.get("model_kwargs", {})).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            mask = batch["mask"].to(device)
            logits = model(x, mask)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(batch["label"].numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    metrics = classification_metrics(labels, preds)
    ci_low, ci_high = confidence_interval_binomial(metrics["accuracy"], n=len(labels))
    metrics["accuracy_ci_low"], metrics["accuracy_ci_high"] = ci_low, ci_high
    metrics["n_test_samples"] = len(labels)

    tables_out = Path(args.tables_out)
    figures_out = Path(args.figures_out)
    tables_out.mkdir(parents=True, exist_ok=True)
    figures_out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(tables_out / f"{args.split}_results_{args.tag}.csv", index=False)

    cm = confusion(labels, preds, num_classes=len(class_names))
    plt.figure(figsize=(max(6, len(class_names) * 0.3), max(5, len(class_names) * 0.3)))
    sns.heatmap(cm, cmap="Blues", cbar=True,
                xticklabels=class_names if len(class_names) <= 40 else False,
                yticklabels=class_names if len(class_names) <= 40 else False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix ({args.split}, {args.tag})")
    plt.tight_layout()
    plt.savefig(figures_out / f"confusion_matrix_{args.tag}.png", dpi=150)
    plt.close()

    pcf1 = per_class_f1(labels, preds, class_names)
    pcf1_series = pd.Series(pcf1).sort_values()
    plt.figure(figsize=(8, max(4, len(class_names) * 0.25)))
    pcf1_series.plot(kind="barh")
    plt.xlabel("F1")
    plt.title(f"Per-class F1 ({args.split}, {args.tag})")
    plt.tight_layout()
    plt.savefig(figures_out / f"per_class_f1_{args.tag}.png", dpi=150)
    plt.close()

    print(f"[evaluate] split={args.split} tag={args.tag} "
          f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
          f"n={metrics['n_test_samples']} -> {tables_out}")


if __name__ == "__main__":
    main()
