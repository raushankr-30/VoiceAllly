"""STEP 10-11: MediaPipe Holistic landmark extraction + caching (spec
section 8).

For every video in the (selected-vocabulary, split-assigned) metadata,
runs MediaPipe Holistic once, saves the FULL raw landmark sequence
(hands + full pose + full face) as a single .npz per video under
data/processed/landmarks_raw/<video_id>.npz. Downstream code (normalize
+ landmark_spec subsetting + temporal resampling) is applied at
train/eval time from this cache -- extraction itself is only ever run
once per video (spec: "do not repeatedly run landmark extraction during
model training").

Usage:
    python src/features/extract_landmarks.py --mapping configs/column_mapping.yaml \
        --raw-dir data/raw --out-dir data/processed/landmarks_raw
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def extract_one_video(video_path: Path, model_path: Path | None = None,
                      frame_stride: int = 1):
    """Runs MediaPipe Holistic frame-by-frame over one video.
    Returns dict of arrays: hand_left (T,21,3), hand_right (T,21,3),
    pose (T,33,3), face (T,468,3), each NaN where MediaPipe did not
    detect that component on a given frame (recorded, not imputed here).
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    left_hand, right_hand, pose, face = [], [], [], []
    def to_arr(landmarks, n):
        """Return a fixed-size xyz array; retain missing data as NaN."""
        arr = np.full((n, 3), np.nan, dtype=np.float32)
        if landmarks is None:
            return arr
        points = getattr(landmarks, "landmark", landmarks)
        if not points:
            return arr
        values = np.array([[lm.x, lm.y, lm.z] for lm in points], dtype=np.float32)
        arr[:min(n, len(values))] = values[:n]
        return arr

    # MediaPipe <=0.10.21 exposes the legacy Solutions API. Newer wheels
    # (including the Python 3.13 compatible build) expose Tasks only.
    use_tasks = not hasattr(mp, "solutions")
    if use_tasks:
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(
                "MediaPipe Tasks requires --model-path pointing to the official "
                "holistic_landmarker.task model bundle."
            )
        options = mp.tasks.vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        detector = mp.tasks.vision.HolisticLandmarker.create_from_options(options)
    else:
        detector = mp.solutions.holistic.Holistic(static_image_mode=False,
                                                   model_complexity=1,
                                                   smooth_landmarks=True)

    try:
        frame_index = 0
        source_frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if source_frame_index % frame_stride:
                source_frame_index += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if use_tasks:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = detector.detect_for_video(image, frame_index * 33)
                left = results.left_hand_landmarks
                right = results.right_hand_landmarks
                body = results.pose_landmarks
                facial = results.face_landmarks
            else:
                results = detector.process(rgb)
                left = results.left_hand_landmarks
                right = results.right_hand_landmarks
                body = results.pose_landmarks
                facial = results.face_landmarks
            left_hand.append(to_arr(left, 21))
            right_hand.append(to_arr(right, 21))
            pose.append(to_arr(body, 33))
            face.append(to_arr(facial, 468))
            frame_index += 1
            source_frame_index += 1
    finally:
        detector.close()
    cap.release()

    if len(pose) == 0:
        raise ValueError(f"No frames read from {video_path}")

    return {
        "left_hand": np.stack(left_hand),
        "right_hand": np.stack(right_hand),
        "pose": np.stack(pose),
        "face": np.stack(face),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="configs/column_mapping.yaml")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out-dir", default="data/processed/landmarks_raw")
    ap.add_argument("--stats-out", default="results/tables/landmark_extraction_statistics.csv")
    ap.add_argument("--model-path", default="assets/mediapipe/holistic_landmarker.task",
                    help="Official MediaPipe Tasks model bundle; required by recent MediaPipe versions.")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="Process every Nth decoded frame (1 keeps the native frame rate).")
    args = ap.parse_args()

    mapping = yaml.safe_load(open(args.mapping))
    video_col = mapping["video_path_col"]
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = Path(args.splits_dir)
    all_rows = pd.concat([
        pd.read_csv(splits_dir / "train.csv"),
        pd.read_csv(splits_dir / "validation.csv"),
        pd.read_csv(splits_dir / "test.csv"),
    ], ignore_index=True)

    stats = []
    for _, row in tqdm(all_rows.iterrows(), total=len(all_rows), desc="extracting landmarks"):
        video_rel = row[video_col]
        video_path = Path(video_rel) if Path(video_rel).is_absolute() else raw_dir / video_rel
        video_id = Path(video_rel).stem
        out_path = out_dir / f"{video_id}.npz"

        if out_path.exists():
            stats.append({"video_id": video_id, "status": "cached", "num_frames": None, "seconds": 0.0})
            continue

        t0 = time.time()
        try:
            seqs = extract_one_video(video_path, Path(args.model_path), args.frame_stride)
            np.savez_compressed(out_path, **seqs)
            stats.append({
                "video_id": video_id,
                "status": "success",
                "num_frames": seqs["pose"].shape[0],
                "seconds": time.time() - t0,
            })
        except Exception as e:  # noqa: BLE001 - we want to record and continue, not crash the whole run
            stats.append({
                "video_id": video_id,
                "status": f"failed: {e}",
                "num_frames": None,
                "seconds": time.time() - t0,
            })

    stats_df = pd.DataFrame(stats)
    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(args.stats_out, index=False)

    n_ok = (stats_df["status"].isin(["success", "cached"])).sum()
    n_fail = (stats_df["status"].str.startswith("failed")).sum()
    print(f"[extract] {n_ok} ok, {n_fail} failed, {len(stats_df)} total -> {args.stats_out}")
    if n_fail:
        print("[extract] failed videos are listed in the stats CSV with their error; "
              "not silently dropped.")


if __name__ == "__main__":
    main()
