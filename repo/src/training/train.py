"""STEP 14-17: Single-config train + validate entrypoint (spec section 13).

Trains one (representation, feature_set, model) combination on TRAIN,
selects the best checkpoint by VALIDATION macro-F1 (never test), and
writes everything an experiment needs for reproducibility: config
snapshot, seed, training history, checkpoint. The TEST set is never
touched by this script -- see src/training/evaluate.py for that, run
only once per spec section 18/26.

Usage:
    python src/training/train.py --config configs/E02_modality_R2.yaml \
        --exp-dir experiments/E02_modality/R2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.features.dataset import LandmarkSequenceDataset, expected_feature_dim
from src.models.architectures import build_model, count_parameters
from src.utils.config import load_config, save_config_snapshot
from src.utils.metrics import classification_metrics
from src.utils.seed import set_seed


def build_label_maps(vocab_csv: str, label_col_name: str = "class_id"):
    vocab = pd.read_csv(vocab_csv)
    selected = vocab.loc[vocab["selected"], label_col_name].tolist()
    label_to_idx = {label: i for i, label in enumerate(sorted(selected))}
    return label_to_idx


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, all_preds, all_labels = 0.0, [], []
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch in loader:
        x = batch["features"].to(device)
        mask = batch["mask"].to(device)
        y = batch["label"].to(device)

        if train:
            optimizer.zero_grad()
        logits = model(x, mask)
        loss = loss_fn(logits, y)
        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        all_preds.append(logits.argmax(dim=-1).detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    metrics = classification_metrics(labels, preds)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp-dir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_to_idx = build_label_maps(cfg["vocabulary_csv"])
    num_classes = len(label_to_idx)

    common = dict(
        landmarks_dir=cfg["landmarks_dir"], video_col=cfg["video_col"], label_col=cfg["label_col"],
        label_to_idx=label_to_idx, representation=cfg["representation"], feature_set=cfg["feature_set"],
        sequence_length=cfg["sequence_length"],
    )
    train_ds = LandmarkSequenceDataset(split_csv=cfg["train_csv"], **common)
    val_ds = LandmarkSequenceDataset(split_csv=cfg["val_csv"], **common)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 0))

    input_dim = expected_feature_dim(cfg["representation"], cfg["feature_set"])
    model = build_model(cfg["model"], input_dim=input_dim, num_classes=num_classes,
                         **cfg.get("model_kwargs", {})).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    save_config_snapshot(cfg, exp_dir)
    n_params = count_parameters(model)

    history = []
    best_val_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improve = 0
    patience = cfg.get("early_stopping_patience", 8)

    t_start = time.time()
    for epoch in range(cfg["epochs"]):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"[train] epoch={epoch} train_loss={train_metrics['loss']:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(model.state_dict(), exp_dir / "checkpoint_best.pt")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"[train] early stopping at epoch {epoch} (no val_macro_f1 improvement for {patience} epochs)")
                break

    total_train_time = time.time() - t_start

    pd.DataFrame(history).to_csv(exp_dir / "training_history.csv", index=False)
    (exp_dir / "label_to_idx.json").write_text(json.dumps(label_to_idx, indent=2))
    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "num_parameters": n_params,
        "num_classes": num_classes,
        "input_dim": input_dim,
        "training_time_seconds": total_train_time,
    }
    (exp_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] done. best_epoch={best_epoch} best_val_macro_f1={best_val_macro_f1:.4f} "
          f"params={n_params} -> {exp_dir}")


if __name__ == "__main__":
    main()
