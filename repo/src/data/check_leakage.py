"""STEP 9: PASS/FAIL leakage report (spec section 7).

Checks:
  - signer overlap between train/validation/test (if signer_col available)
  - duplicate video files (by content hash) across splits

Exits with non-zero status on FAIL so it can be used as a CI-style gate
before any training script is allowed to run.

Usage:
    python src/data/check_leakage.py --mapping configs/column_mapping.yaml \
        --splits-dir data/splits --raw-dir data/raw
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import yaml


def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="configs/column_mapping.yaml")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args()

    mapping = yaml.safe_load(open(args.mapping))
    signer_col = mapping.get("signer_col")
    video_col = mapping["video_path_col"]
    splits_dir = Path(args.splits_dir)
    raw_dir = Path(args.raw_dir)

    train = pd.read_csv(splits_dir / "train.csv")
    val = pd.read_csv(splits_dir / "validation.csv")
    test = pd.read_csv(splits_dir / "test.csv")
    split_mode = (splits_dir / "SPLIT_MODE.txt").read_text().strip() if (splits_dir / "SPLIT_MODE.txt").exists() else "unknown"

    lines = [f"Split mode: {split_mode}", ""]
    status_ok = True

    if signer_col and signer_col in train.columns:
        tr_s, va_s, te_s = set(train[signer_col]), set(val[signer_col]), set(test[signer_col])
        tv = len(tr_s & va_s)
        tt = len(tr_s & te_s)
        vt = len(va_s & te_s)
        lines += [
            f"TRAIN/VALIDATION signer overlap: {tv}",
            f"TRAIN/TEST signer overlap: {tt}",
            f"VALIDATION/TEST signer overlap: {vt}",
        ]
        if tv or tt or vt:
            status_ok = False
    else:
        lines.append("Signer overlap check: SKIPPED (no signer_col in mapping -- "
                      "this split is not signer-independent; see SPLIT_MODE.txt)")

    # Duplicate-file check by content hash across splits
    def hashes_for(df: pd.DataFrame) -> set[str]:
        out = set()
        for p in df[video_col]:
            full = Path(p) if Path(p).is_absolute() else raw_dir / p
            if full.exists():
                out.add(file_hash(full))
        return out

    h_tr, h_va, h_te = hashes_for(train), hashes_for(val), hashes_for(test)
    dup = len((h_tr & h_va) | (h_tr & h_te) | (h_va & h_te))
    lines.append(f"Duplicate file overlap (content hash): {dup}")
    if dup:
        status_ok = False

    lines.append(f"STATUS: {'PASS' if status_ok else 'FAIL'}")
    report = "\n".join(lines)
    print(report)

    report_path = splits_dir / "LEAKAGE_REPORT.txt"
    report_path.write_text(report + "\n")
    if not status_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
