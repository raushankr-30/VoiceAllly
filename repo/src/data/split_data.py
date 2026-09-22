"""STEP 8: Leakage-controlled, group-aware (signer-independent) splitting.

Per spec section 7: if reliable signer IDs exist, split BY SIGNER so no
signer appears in more than one of train/validation/test. If signer IDs
are not available/reliable, this script refuses to fabricate a
signer-independent split and exits with a clear message instead --
it will optionally fall back to a plain (non-signer-independent)
video-level split ONLY if --allow-non-signer-independent is passed
explicitly, and it will label the resulting splits as such everywhere
downstream.

Usage:
    python src/data/split_data.py --mapping configs/column_mapping.yaml \
        --vocabulary data/processed/vocabulary.csv \
        --val-size 0.15 --test-size 0.15 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit


def group_aware_split(df: pd.DataFrame, group_col: str, val_size: float,
                       test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=df[group_col]))
    trainval_df, test_df = df.iloc[trainval_idx], df.iloc[test_idx]

    relative_val = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_df[group_col]))
    train_df, val_df = trainval_df.iloc[train_idx], trainval_df.iloc[val_idx]

    return train_df, val_df, test_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="configs/column_mapping.yaml")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--vocabulary", default="data/processed/vocabulary.csv")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--allow-non-signer-independent", action="store_true",
                     help="Explicit opt-in to fall back to a random video-level split when "
                          "signer metadata is unavailable. The resulting splits are labeled "
                          "'video_level_random' (NOT signer-independent) everywhere downstream.")
    args = ap.parse_args()

    mapping = yaml.safe_load(open(args.mapping))
    signer_col = mapping.get("signer_col")
    label_col = mapping["label_col"]
    video_col = mapping["video_path_col"]

    raw_df = pd.read_csv(Path(args.raw_dir) / mapping["primary_metadata_file"])
    vocab = pd.read_csv(args.vocabulary)
    selected_labels = set(vocab.loc[vocab["selected"], "class_id"])
    df = raw_df[raw_df[label_col].isin(selected_labels)].reset_index(drop=True)

    signer_available = signer_col is not None and signer_col in df.columns and df[signer_col].notna().all()

    if signer_available:
        split_mode = "signer_independent"
        train_df, val_df, test_df = group_aware_split(
            df, signer_col, args.val_size, args.test_size, args.seed
        )
    else:
        if not args.allow_non_signer_independent:
            raise SystemExit(
                "[split] Reliable signer IDs are not available in the mapped metadata. "
                "Per spec section 7, I will NOT fabricate a signer-independent split. "
                "If you have confirmed there is genuinely no usable signer grouping in CISLR "
                "and still want a baseline split for a documented limitation, re-run with "
                "--allow-non-signer-independent -- the resulting splits will be explicitly "
                "labeled 'video_level_random', not signer-independent, and "
                "research/DATASET_CARD.md must state this limitation."
            )
        split_mode = "video_level_random"
        rng = np.random.RandomState(args.seed)
        idx = rng.permutation(len(df))
        n_test = int(len(df) * args.test_size)
        n_val = int(len(df) * args.val_size)
        test_df = df.iloc[idx[:n_test]]
        val_df = df.iloc[idx[n_test:n_test + n_val]]
        train_df = df.iloc[idx[n_test + n_val:]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "validation.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    (out_dir / "SPLIT_MODE.txt").write_text(split_mode + "\n")

    print(f"[split] mode={split_mode} train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"-> {out_dir}")
    print("[split] Run src/data/check_leakage.py next to verify group disjointness before "
          "any extraction/training uses these splits.")


if __name__ == "__main__":
    main()
