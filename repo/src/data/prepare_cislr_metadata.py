"""Build a reproducible labeled manifest for the locally delivered CISLR subset.

The official CSVs identify a video by ``uid`` while the local video files are
named ``<uid>.mp4`` under ``data/raw/Data_1500``. This script records that
join explicitly; it never infers labels or signer IDs from filenames.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/raw/local_manifest.csv")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    dataset = pd.read_csv(raw_dir / "dataset.csv")
    prototype = set(pd.read_csv(raw_dir / "prototype.csv")["uid"])
    test = set(pd.read_csv(raw_dir / "test.csv")["uid"])

    # Prefer the complete official archive when it has been extracted; retain
    # support for the previously delivered 1,500-video pilot subset.
    video_dir = raw_dir / "CISLR_v1.5-a_videos"
    if not video_dir.exists():
        video_dir = raw_dir / "Data_1500"
    available = {p.stem for p in video_dir.glob("*.mp4")}
    manifest = dataset[dataset["uid"].isin(available)].copy()
    manifest["video_path"] = video_dir.name + "/" + manifest["uid"] + ".mp4"
    manifest["official_partition"] = manifest["uid"].map(
        lambda uid: "prototype" if uid in prototype else "test" if uid in test else "unassigned"
    )
    if (manifest["official_partition"] == "unassigned").any():
        raise RuntimeError("A local video has no official prototype/test assignment.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)
    print({
        "local_labeled_videos": len(manifest),
        "local_glosses": manifest["gloss"].nunique(),
        "prototype": int((manifest["official_partition"] == "prototype").sum()),
        "test": int((manifest["official_partition"] == "test").sum()),
        "output": args.out,
    })


if __name__ == "__main__":
    main()
