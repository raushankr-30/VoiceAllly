"""STEP 16 feature ablation: position / velocity / acceleration (spec
section 16), computed AFTER normalization and AFTER fixed-length
resampling, so velocity/acceleration are computed on a consistent
frame spacing across the whole dataset.

v_t = p_t - p_(t-1), a_t = v_t - v_(t-1), exactly as specified. First
frame's velocity and first two frames' acceleration are zero-filled
(defined, not fabricated -- documented here).
"""
from __future__ import annotations

import numpy as np


def add_velocity(seq: np.ndarray) -> np.ndarray:
    """seq: (T, D) position features. Returns (T, D) velocity, v[0] = 0."""
    v = np.zeros_like(seq)
    v[1:] = seq[1:] - seq[:-1]
    return v


def add_acceleration(vel: np.ndarray) -> np.ndarray:
    """vel: (T, D) velocity features. Returns (T, D) acceleration, a[0]=a[1]=0."""
    a = np.zeros_like(vel)
    a[1:] = vel[1:] - vel[:-1]
    return a


def build_feature_set(position_seq: np.ndarray, feature_set: str) -> np.ndarray:
    """feature_set: 'F1' (position only), 'F2' (position+velocity),
    'F3' (position+velocity+acceleration). Concatenated along the
    feature (last) axis.
    """
    if feature_set == "F1":
        return position_seq
    vel = add_velocity(position_seq)
    if feature_set == "F2":
        return np.concatenate([position_seq, vel], axis=-1)
    if feature_set == "F3":
        acc = add_acceleration(vel)
        return np.concatenate([position_seq, vel, acc], axis=-1)
    raise ValueError(f"Unknown feature_set: {feature_set}")
