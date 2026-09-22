"""STEP 12: Translation/scale normalization (spec section 10).

Reference point: midpoint of the two shoulders (pose indices 11, 12),
which is stable across frames even when hands move fast and is present
whenever pose landmarks are extracted.

Scale: the shoulder-to-shoulder distance in the first valid frame of the
sequence (a body-size-invariant, signer-invariant scale). This exact
choice is recorded here and must not silently change between experiments
per spec section 10 -- if you change it, update PREPROCESSING_VERSION.
"""
from __future__ import annotations

import numpy as np

PREPROCESSING_VERSION = "norm_v1_shoulder_midpoint_shoulder_width"

LEFT_SHOULDER_POSE_IDX = 11
RIGHT_SHOULDER_POSE_IDX = 12


def compute_reference_and_scale(pose_xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """pose_xyz: (T, num_pose_landmarks, 3) raw pose coordinates for the
    full BlazePose topology (indices as extracted, before any subsetting).
    Returns (reference_point (3,), scale (scalar)).
    """
    valid_frames = ~np.isnan(pose_xyz[:, LEFT_SHOULDER_POSE_IDX, 0])
    if not valid_frames.any():
        raise ValueError("No frame has valid shoulder landmarks; cannot normalize this sequence.")
    first_valid = np.argmax(valid_frames)
    left = pose_xyz[first_valid, LEFT_SHOULDER_POSE_IDX]
    right = pose_xyz[first_valid, RIGHT_SHOULDER_POSE_IDX]
    reference = (left + right) / 2.0
    scale = float(np.linalg.norm(left - right))
    if scale < 1e-6:
        raise ValueError("Degenerate shoulder-width scale (~0); check landmark extraction for this video.")
    return reference, scale


def normalize_sequence(points_xyz: np.ndarray, reference: np.ndarray, scale: float) -> np.ndarray:
    """points_xyz: (T, N, 3) any landmark group (hands/pose/face) for one
    sequence. Applies p_normalized = (p - reference) / scale, per spec
    section 10's exact formula.
    """
    return (points_xyz - reference[None, None, :]) / scale
