"""Turns cached raw landmark .npz files (from extract_landmarks.py) into
fixed-length, normalized, representation- and feature-set-selected
tensors, on the fly, for a given (representation, feature_set,
sequence_length). This is where R1/R2/R3 and F1/F2/F3 are actually
applied -- extraction itself never needs to be re-run to switch these.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.kinematics import build_feature_set
from src.features.landmark_spec import (
    FACE_SELECTED_INDICES,
    POSE_UPPER_BODY_INDICES,
    REPRESENTATIONS,
    feature_dim,
)
from src.features.normalize import compute_reference_and_scale, normalize_sequence
from src.features.temporal import resample_or_pad


class LandmarkSequenceDataset(Dataset):
    def __init__(self, split_csv: str, landmarks_dir: str, video_col: str, label_col: str,
                 label_to_idx: dict, representation: str, feature_set: str, sequence_length: int):
        self.df = pd.read_csv(split_csv)
        self.landmarks_dir = Path(landmarks_dir)
        self.video_col = video_col
        self.label_col = label_col
        self.label_to_idx = label_to_idx
        self.representation = representation
        self.feature_set = feature_set
        self.sequence_length = sequence_length
        self.repr_spec = REPRESENTATIONS[representation]

    def __len__(self):
        return len(self.df)

    def _build_position_sequence(self, npz) -> np.ndarray:
        parts = []
        if self.repr_spec["uses_hands"]:
            parts.append(npz["left_hand"].reshape(npz["left_hand"].shape[0], -1))
            parts.append(npz["right_hand"].reshape(npz["right_hand"].shape[0], -1))
        if self.repr_spec["uses_pose"]:
            reference, scale = compute_reference_and_scale(npz["pose"])
            pose_sel = npz["pose"][:, POSE_UPPER_BODY_INDICES, :]
            pose_sel = normalize_sequence(pose_sel, reference, scale)
            parts.append(pose_sel.reshape(pose_sel.shape[0], -1))
            if self.repr_spec["uses_face"]:
                face_sel = npz["face"][:, FACE_SELECTED_INDICES, :]
                face_sel = normalize_sequence(face_sel, reference, scale)
                parts.append(face_sel.reshape(face_sel.shape[0], -1))
            # hands are normalized with the same reference/scale for consistency
            if self.repr_spec["uses_hands"]:
                lh = normalize_sequence(npz["left_hand"], reference, scale).reshape(npz["left_hand"].shape[0], -1)
                rh = normalize_sequence(npz["right_hand"], reference, scale).reshape(npz["right_hand"].shape[0], -1)
                parts = [lh, rh] + parts[2:] if self.repr_spec["uses_pose"] else [lh, rh]
        elif self.repr_spec["uses_hands"]:
            # R1: no pose available for a body-relative reference. Fall back to a
            # hands-only reference: midpoint of both wrists in the first valid frame.
            wrists = np.concatenate([npz["left_hand"][:, 0:1], npz["right_hand"][:, 0:1]], axis=1)
            # A signer can use one hand only. Choose the first frame where
            # either wrist is observed, then average only observed wrists.
            valid = ~np.isnan(wrists).all(axis=(1, 2))
            first_valid = int(np.argmax(valid)) if valid.any() else 0
            observed_wrists = wrists[first_valid][~np.isnan(wrists[first_valid]).any(axis=1)]
            reference = observed_wrists.mean(axis=0) if len(observed_wrists) else np.zeros(3, dtype=np.float32)
            left, right = npz["left_hand"][first_valid], npz["right_hand"][first_valid]
            if not np.isnan(left[0]).any() and not np.isnan(right[0]).any():
                scale = float(np.linalg.norm(left[0] - right[0]))
            elif not np.isnan(left[[0, 9]]).any():
                scale = float(np.linalg.norm(left[0] - left[9]))
            elif not np.isnan(right[[0, 9]]).any():
                scale = float(np.linalg.norm(right[0] - right[9]))
            else:
                scale = 1.0
            scale = scale if np.isfinite(scale) and scale > 1e-6 else 1.0
            lh = normalize_sequence(npz["left_hand"], reference, scale).reshape(npz["left_hand"].shape[0], -1)
            rh = normalize_sequence(npz["right_hand"], reference, scale).reshape(npz["right_hand"].shape[0], -1)
            parts = [lh, rh]

        seq = np.concatenate(parts, axis=-1)
        seq = np.nan_to_num(seq, nan=0.0)
        return seq.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = Path(row[self.video_col]).stem
        npz = np.load(self.landmarks_dir / f"{video_id}.npz")

        pos_seq = self._build_position_sequence(npz)
        fixed, mask = resample_or_pad(pos_seq, self.sequence_length)
        features = build_feature_set(fixed, self.feature_set)

        label = self.label_to_idx[row[self.label_col]]
        return {
            "features": torch.from_numpy(features),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label, dtype=torch.long),
        }


def expected_feature_dim(representation: str, feature_set: str) -> int:
    base = feature_dim(representation)
    multiplier = {"F1": 1, "F2": 2, "F3": 3}[feature_set]
    return base * multiplier
