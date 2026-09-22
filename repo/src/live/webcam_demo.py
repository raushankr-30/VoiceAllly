"""STEP 23: Live webcam validation (spec section 24). This is only a
practical sanity check of the frozen final model on live signers not in
training -- deliberately no UI beyond a plain OpenCV window with text
overlay, and results are logged to results/live_test.csv exactly as
specified. Must be run on a machine with a physical camera (not this
sandbox); nothing here should be treated as passing/failing evidence
until it has actually been run and logged.

Usage (on a machine with a webcam):
    python src/live/webcam_demo.py --config experiments/E07_final/config.yaml \
        --checkpoint experiments/E07_final/checkpoint_best.pt --signer-name alice
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch

from src.features.dataset import expected_feature_dim
from src.features.kinematics import build_feature_set
from src.features.landmark_spec import FACE_SELECTED_INDICES, POSE_UPPER_BODY_INDICES, REPRESENTATIONS
from src.features.normalize import compute_reference_and_scale, normalize_sequence
from src.features.temporal import resample_or_pad
from src.models.architectures import build_model
from src.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--signer-name", required=True, help="Label for the person testing, for results/live_test.csv")
    ap.add_argument("--window-seconds", type=float, default=2.0, help="Rolling buffer length fed to the model.")
    ap.add_argument("--confidence-threshold", type=float, default=0.6)
    ap.add_argument("--out", default="results/live_test.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu")  # webcam demo prioritizes portability over speed
    exp_dir = Path(args.checkpoint).parent
    label_to_idx = json.loads((exp_dir / "label_to_idx.json").read_text())
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    input_dim = expected_feature_dim(cfg["representation"], cfg["feature_set"])
    model = build_model(cfg["model"], input_dim=input_dim, num_classes=len(label_to_idx),
                         **cfg.get("model_kwargs", {})).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    repr_spec = REPRESENTATIONS[cfg["representation"]]
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("No webcam detected. This script must be run on a machine with a camera.")

    buffer_lh, buffer_rh, buffer_pose, buffer_face = [], [], [], []
    fps_estimate = 25.0
    buffer_len = int(args.window_seconds * fps_estimate)
    logs = []

    print("[live] press 'q' to quit, 'l' to log the last prediction as correct/incorrect for a class.")
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            def to_arr(landmarks, n):
                if landmarks is None:
                    return np.full((n, 3), np.nan, dtype=np.float32)
                return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)

            buffer_lh.append(to_arr(results.left_hand_landmarks, 21))
            buffer_rh.append(to_arr(results.right_hand_landmarks, 21))
            buffer_pose.append(to_arr(results.pose_landmarks, 33))
            buffer_face.append(to_arr(results.face_landmarks, 468))
            buffer_lh, buffer_rh, buffer_pose, buffer_face = (
                buffer_lh[-buffer_len:], buffer_rh[-buffer_len:], buffer_pose[-buffer_len:], buffer_face[-buffer_len:]
            )

            pred_label, confidence = None, 0.0
            t_infer_start = time.perf_counter()
            if len(buffer_pose) >= 5:
                pose_arr = np.stack(buffer_pose)
                reference, scale = compute_reference_and_scale(pose_arr)
                parts = []
                if repr_spec["uses_hands"]:
                    parts.append(normalize_sequence(np.stack(buffer_lh), reference, scale).reshape(len(buffer_lh), -1))
                    parts.append(normalize_sequence(np.stack(buffer_rh), reference, scale).reshape(len(buffer_rh), -1))
                if repr_spec["uses_pose"]:
                    pose_sel = normalize_sequence(pose_arr[:, POSE_UPPER_BODY_INDICES], reference, scale)
                    parts.append(pose_sel.reshape(pose_sel.shape[0], -1))
                if repr_spec["uses_face"]:
                    face_sel = normalize_sequence(np.stack(buffer_face)[:, FACE_SELECTED_INDICES], reference, scale)
                    parts.append(face_sel.reshape(face_sel.shape[0], -1))
                seq = np.nan_to_num(np.concatenate(parts, axis=-1), nan=0.0).astype(np.float32)
                fixed, mask = resample_or_pad(seq, cfg["sequence_length"])
                features = build_feature_set(fixed, cfg["feature_set"])
                x = torch.from_numpy(features).unsqueeze(0)
                m = torch.from_numpy(mask).unsqueeze(0)
                with torch.no_grad():
                    logits = model(x, m)
                    probs = torch.softmax(logits, dim=-1)[0]
                    top_idx = int(probs.argmax())
                    confidence = float(probs[top_idx])
                    pred_label = idx_to_label[top_idx] if confidence >= args.confidence_threshold else "REJECTED(low_confidence)"
            latency_ms = (time.perf_counter() - t_infer_start) * 1000

            overlay = f"{pred_label} ({confidence:.2f}) [{latency_ms:.0f}ms]" if pred_label else "..."
            cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("VoiceAlly live validation", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l") and pred_label is not None:
                true_class = input("True class for this attempt: ").strip()
                correct = input("Was the prediction correct? (y/n): ").strip().lower() == "y"
                logs.append({
                    "signer": args.signer_name, "class": true_class, "attempt": len(logs) + 1,
                    "prediction": pred_label, "confidence": confidence,
                    "correct": correct, "latency_ms": latency_ms,
                })
                print(f"[live] logged attempt {len(logs)}")

    cap.release()
    cv2.destroyAllWindows()

    if logs:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(logs)
        if out_path.exists():
            df = pd.concat([pd.read_csv(out_path), df], ignore_index=True)
        df.to_csv(out_path, index=False)
        print(f"[live] {len(logs)} attempts logged -> {out_path}")
    else:
        print("[live] no attempts logged this session.")


if __name__ == "__main__":
    main()
