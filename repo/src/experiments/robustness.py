"""STEP 21: Limited, clearly-defined robustness evaluation (spec section
20). Honesty note up front: of the five perturbations listed in the
spec, two (lighting variation, background variation) are properties of
the RAW VIDEO, not of already-extracted landmarks, and can only be
tested by re-running MediaPipe extraction on re-encoded/augmented
videos (a separate, more expensive step -- see augment_and_reextract()
stub below, not wired into this script by default). This script
implements the three perturbations that operate validly at the
landmark-sequence level, and reports the other two as NOT YET MEASURED
until video-level augmentation is actually run, rather than faking them
with landmark-level proxies:

  1. signing speed variation  -- time-resample the sequence faster/slower
     before the fixed-length resampling step (genuinely valid at landmark level)
  2. partial occlusion        -- zero out one hand's landmarks for a
     contiguous span of frames (genuinely valid at landmark level)
  3. landmark dropout         -- randomly zero out individual landmark
     points at a given rate (genuinely valid at landmark level)

Usage:
    python src/experiments/robustness.py --config experiments/E07_final/config.yaml \
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
from src.utils.metrics import classification_metrics


def speed_perturb(seq: np.ndarray, factor: float) -> np.ndarray:
    t = seq.shape[0]
    new_t = max(2, int(round(t / factor)))
    idx = np.linspace(0, t - 1, new_t).round().astype(int)
    return seq[idx]


def occlusion_perturb(seq: np.ndarray, hand_dim: int, span_frac: float, rng: np.random.RandomState) -> np.ndarray:
    seq = seq.copy()
    t = seq.shape[0]
    span = max(1, int(t * span_frac))
    start = rng.randint(0, max(1, t - span))
    seq[start:start + span, :hand_dim] = 0.0  # zero out first-hand block
    return seq


def dropout_perturb(seq: np.ndarray, rate: float, rng: np.random.RandomState) -> np.ndarray:
    mask = rng.rand(*seq.shape) < rate
    seq = seq.copy()
    seq[mask] = 0.0
    return seq


PERTURBATIONS = ["speed_variation", "partial_occlusion", "landmark_dropout"]
NOT_MEASURED = ["lighting_variation", "background_variation"]


def evaluate_with_perturbation(model, ds, device, perturb_name: str, rng_seed: int):
    """Runs the model over the dataset applying one perturbation to each
    raw feature sequence before mask/tensor conversion. Re-implements the
    minimal parts of LandmarkSequenceDataset.__getitem__ needed to inject
    the perturbation at the position-sequence stage.
    """
    from src.features.kinematics import build_feature_set
    from src.features.temporal import resample_or_pad

    rng = np.random.RandomState(rng_seed)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            row = ds.df.iloc[i]
            video_id = Path(row[ds.video_col]).stem
            npz = np.load(ds.landmarks_dir / f"{video_id}.npz")
            pos_seq = ds._build_position_sequence(npz)

            if perturb_name == "speed_variation":
                factor = rng.choice([0.75, 1.25])
                pos_seq = speed_perturb(pos_seq, factor)
            elif perturb_name == "partial_occlusion":
                pos_seq = occlusion_perturb(pos_seq, hand_dim=21 * 3, span_frac=0.3, rng=rng)
            elif perturb_name == "landmark_dropout":
                pos_seq = dropout_perturb(pos_seq, rate=0.15, rng=rng)

            fixed, mask = resample_or_pad(pos_seq, ds.sequence_length)
            features = build_feature_set(fixed, ds.feature_set)
            x = torch.from_numpy(features).unsqueeze(0).to(device)
            m = torch.from_numpy(mask).unsqueeze(0).to(device)
            logits = model(x, m)
            all_preds.append(int(logits.argmax(dim=-1).cpu().item()))
            all_labels.append(int(ds.label_to_idx[row[ds.label_col]]))
    return np.array(all_labels), np.array(all_preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="results/tables/robustness.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(args.checkpoint).parent
    label_to_idx = json.loads((exp_dir / "label_to_idx.json").read_text())

    ds = LandmarkSequenceDataset(
        split_csv=cfg["test_csv"], landmarks_dir=cfg["landmarks_dir"], video_col=cfg["video_col"],
        label_col=cfg["label_col"], label_to_idx=label_to_idx, representation=cfg["representation"],
        feature_set=cfg["feature_set"], sequence_length=cfg["sequence_length"],
    )
    input_dim = expected_feature_dim(cfg["representation"], cfg["feature_set"])
    model = build_model(cfg["model"], input_dim=input_dim, num_classes=len(label_to_idx),
                         **cfg.get("model_kwargs", {})).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Clean baseline
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False)
    clean_preds, clean_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            m = batch["mask"].to(device)
            logits = model(x, m)
            clean_preds.append(logits.argmax(dim=-1).cpu().numpy())
            clean_labels.append(batch["label"].numpy())
    clean_preds = np.concatenate(clean_preds)
    clean_labels = np.concatenate(clean_labels)
    clean_metrics = classification_metrics(clean_labels, clean_preds)

    rows = [{"condition": "clean", **clean_metrics,
             "absolute_degradation_macro_f1": 0.0, "relative_degradation_macro_f1_pct": 0.0}]

    for cond in PERTURBATIONS:
        labels, preds = evaluate_with_perturbation(model, ds, device, cond, rng_seed=args.seed)
        m = classification_metrics(labels, preds)
        abs_deg = clean_metrics["macro_f1"] - m["macro_f1"]
        rel_deg = (abs_deg / clean_metrics["macro_f1"] * 100) if clean_metrics["macro_f1"] > 0 else float("nan")
        rows.append({"condition": cond, **m,
                      "absolute_degradation_macro_f1": abs_deg,
                      "relative_degradation_macro_f1_pct": rel_deg})

    for cond in NOT_MEASURED:
        rows.append({"condition": cond, "accuracy": "NOT YET MEASURED", "macro_f1": "NOT YET MEASURED",
                     "weighted_f1": "NOT YET MEASURED", "precision_macro": "NOT YET MEASURED",
                     "recall_macro": "NOT YET MEASURED",
                     "absolute_degradation_macro_f1": "NOT YET MEASURED",
                     "relative_degradation_macro_f1_pct": "NOT YET MEASURED"})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[robustness] wrote {len(rows)} conditions -> {out_path} "
          f"({len(NOT_MEASURED)} require video-level augmentation, not yet implemented)")


if __name__ == "__main__":
    main()
