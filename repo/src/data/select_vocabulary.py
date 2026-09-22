"""STEP 6: Reproducible vocabulary-selection rule (spec section 6).

Reads results/tables/dataset_statistics.csv-level info directly from the
mapped metadata (not from a hardcoded class count) and applies an explicit,
documented rule:

  A class is SELECTED iff:
    - it has >= min_videos_per_class usable videos, AND
    - (if signer metadata exists) it has >= min_signers_per_class distinct
      signers, AND
    - its videos are not exclusively missing/corrupt files.

Everything excluded gets a reason. Nothing is picked because a round
number "sounds impressive" (spec explicitly forbids this).

Usage:
    python src/data/select_vocabulary.py --mapping configs/column_mapping.yaml \
        --min-videos-per-class 15 --min-signers-per-class 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def select_vocabulary(df: pd.DataFrame, mapping: dict,
                       min_videos_per_class: int,
                       min_signers_per_class: int,
                       raw_dir: Path) -> pd.DataFrame:
    label_col = mapping["label_col"]
    signer_col = mapping.get("signer_col")
    video_col = mapping["video_path_col"]

    df = df.copy()
    df["_exists"] = df[video_col].apply(
        lambda p: (Path(p) if Path(p).is_absolute() else raw_dir / p).exists()
    )
    usable = df[df["_exists"]]

    rows = []
    for label, group in df.groupby(label_col):
        usable_group = group[group["_exists"]]
        n_videos = len(usable_group)
        n_signers = usable_group[signer_col].nunique() if signer_col and signer_col in df.columns else None

        reasons = []
        if n_videos < min_videos_per_class:
            reasons.append(f"only {n_videos} usable videos (< {min_videos_per_class})")
        if signer_col and n_signers is not None and n_signers < min_signers_per_class:
            reasons.append(f"only {n_signers} distinct signers (< {min_signers_per_class})")

        selected = len(reasons) == 0
        rows.append({
            "class_id": label,
            "label": label,
            "number_of_videos": n_videos,
            "number_of_signers": n_signers if n_signers is not None else "unavailable",
            "selected": selected,
            "reason_if_excluded": "; ".join(reasons) if reasons else "",
        })

    return pd.DataFrame(rows).sort_values("number_of_videos", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="configs/column_mapping.yaml")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--min-videos-per-class", type=int, default=15,
                     help="Minimum usable videos for a class to be trainable/testable at all "
                          "with a held-out signer-independent split. Adjust once real per-class "
                          "counts are known -- this default is a starting point, not a finding.")
    ap.add_argument("--min-signers-per-class", type=int, default=3,
                     help="Minimum distinct signers per class so a class can be split across "
                          "train/val/test signer groups without a class disappearing from a split.")
    ap.add_argument("--out", default="data/processed/vocabulary.csv")
    args = ap.parse_args()

    mapping = yaml.safe_load(open(args.mapping))
    if mapping.get("label_col") is None or mapping.get("primary_metadata_file") is None:
        raise SystemExit(
            "configs/column_mapping.yaml is not filled in yet. Run audit_dataset.py first, "
            "read the schema report, and set label_col / video_path_col / signer_col."
        )

    raw_dir = Path(args.raw_dir)
    df = pd.read_csv(raw_dir / mapping["primary_metadata_file"])

    vocab = select_vocabulary(
        df, mapping,
        min_videos_per_class=args.min_videos_per_class,
        min_signers_per_class=args.min_signers_per_class,
        raw_dir=raw_dir,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vocab.to_csv(args.out, index=False)

    n_selected = int(vocab["selected"].sum())
    print(f"[vocab] {n_selected}/{len(vocab)} classes selected "
          f"(min_videos_per_class={args.min_videos_per_class}, "
          f"min_signers_per_class={args.min_signers_per_class}) -> {args.out}")


if __name__ == "__main__":
    main()
