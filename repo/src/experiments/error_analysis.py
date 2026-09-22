"""STEP 22: Error analysis on the frozen final model's test predictions
(spec section 23). Identifies the most frequent confusion pairs
statistically; the CAUSE column is left for manual inspection of actual
video samples (spec forbids inventing explanations) -- this script
prepares the ranked pairs and a sample-lookup list so a human (or a
follow-up run of this script with --inspect) can fill it in with
observed causes only.

Usage:
    python src/experiments/error_analysis.py --config experiments/E07_final/config.yaml \
        --checkpoint experiments/E07_final/checkpoint_best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.features.dataset import LandmarkSequenceDataset, expected_feature_dim
from src.models.architectures import build_model
from src.utils.config import load_config
from src.utils.metrics import confusion, top_confusion_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--out", default="results/tables/error_analysis.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(args.checkpoint).parent
    label_to_idx = json.loads((exp_dir / "label_to_idx.json").read_text())
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    ds = LandmarkSequenceDataset(
        split_csv=cfg["test_csv"], landmarks_dir=cfg["landmarks_dir"], video_col=cfg["video_col"],
        label_col=cfg["label_col"], label_to_idx=label_to_idx, representation=cfg["representation"],
        feature_set=cfg["feature_set"], sequence_length=cfg["sequence_length"],
    )
    input_dim = expected_feature_dim(cfg["representation"], cfg["feature_set"])
    model = build_model(cfg["model"], input_dim=input_dim, num_classes=len(class_names),
                         **cfg.get("model_kwargs", {})).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Use batch_size=1 to keep a clean per-sample video_id <-> prediction mapping,
    # since batched shuffling/collation would otherwise decouple predictions from rows.
    loader1 = DataLoader(ds, batch_size=1, shuffle=False)
    all_preds, all_labels, all_video_ids = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader1):
            x = batch["features"].to(device)
            m = batch["mask"].to(device)
            logits = model(x, m)
            all_preds.append(int(logits.argmax(dim=-1).cpu().item()))
            all_labels.append(int(batch["label"].item()))
            all_video_ids.append(Path(ds.df.iloc[i][ds.video_col]).stem)

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    cm = confusion(labels, preds, num_classes=len(class_names))
    pairs = top_confusion_pairs(cm, class_names, top_k=args.top_k)

    rows = []
    for true_label, pred_label, count in pairs:
        example_video_ids = [
            vid for vid, l, p in zip(all_video_ids, labels, preds)
            if class_names[l] == true_label and class_names[p] == pred_label
        ][:5]
        rows.append({
            "true_class": true_label,
            "predicted_class": pred_label,
            "count": count,
            "example_video_ids_for_manual_inspection": ";".join(example_video_ids),
            "observed_cause": "NOT YET INSPECTED",  # fill in only after watching the example videos
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[error_analysis] {len(rows)} confusion pairs -> {out_path}. "
          f"'observed_cause' must be filled in by inspecting the listed example videos "
          f"-- do not guess causes from the confusion count alone.")


if __name__ == "__main__":
    main()
