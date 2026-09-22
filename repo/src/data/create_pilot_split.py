"""Create a tiny, reproducible non-signer-independent pipeline smoke test.

This is intentionally separate from the paper protocol: CISLR has no signer
IDs and too few samples per gloss for the requested main experiment. Each
selected class contributes one validation and one test example; all remaining
examples train the model. Results must be reported only as pilot results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/raw/local_manifest.csv")
    ap.add_argument("--num-classes", type=int, default=5)
    ap.add_argument("--min-videos", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/splits/pilot_non_signer")
    ap.add_argument("--vocabulary-out", default="data/processed/pilot_vocabulary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    counts = df["gloss"].value_counts()
    labels = sorted(counts[counts >= args.min_videos].index,
                    key=lambda label: (-counts[label], label))[:args.num_classes]
    if len(labels) != args.num_classes:
        raise ValueError("Not enough classes meeting the requested minimum.")

    rng = np.random.RandomState(args.seed)
    train, validation, test = [], [], []
    for label in labels:
        group = df[df["gloss"] == label].iloc[rng.permutation(int(counts[label]))]
        test.append(group.iloc[[0]])
        validation.append(group.iloc[[1]])
        train.append(group.iloc[2:])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, parts in [("train", train), ("validation", validation), ("test", test)]:
        pd.concat(parts, ignore_index=True).to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "SPLIT_MODE.txt").write_text(
        "pilot_video_level_stratified; NOT signer-independent; NOT paper result\n"
    )
    pd.DataFrame({
        "class_id": labels, "label": labels,
        "number_of_videos": [int(counts[x]) for x in labels],
        "number_of_signers": "unavailable", "selected": True,
        "reason_if_excluded": "",
    }).to_csv(args.vocabulary_out, index=False)
    print({"labels": labels, "train": sum(len(x) for x in train),
           "validation": len(validation), "test": len(test), "out": str(out_dir)})


if __name__ == "__main__":
    main()
