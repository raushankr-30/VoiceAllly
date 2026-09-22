"""STEP 13: Temporal sampling / padding (spec section 11).

`choose_sequence_length` must be run against the REAL frame-count
distribution of the extracted landmarks (not guessed) -- see
src/features/inspect_sequence_lengths.py. The strategy itself
(uniform resampling to a fixed length + boolean mask) is fixed here so
it is identical across every experiment/representation/model.
"""
from __future__ import annotations

import numpy as np


def choose_sequence_length(frame_counts: np.ndarray, percentile: float = 90.0) -> int:
    """Pick a fixed sequence length at the given percentile of the actual
    observed frame-count distribution (rounded to the nearest 5), so most
    sequences are truncated only slightly and few are padded a lot. This
    must be called on real per-video frame counts, not assumed.
    """
    val = int(np.percentile(frame_counts, percentile))
    return max(5, int(round(val / 5.0)) * 5)


def resample_or_pad(sequence: np.ndarray, target_len: int) -> tuple[np.ndarray, np.ndarray]:
    """sequence: (T, D). Returns (fixed_sequence (target_len, D), mask (target_len,) bool)
    - if T >= target_len: uniformly subsample target_len frame indices (keeps
      motion shape better than naive truncation).
    - if T < target_len: keep all real frames, pad the rest with zeros and
      mask them out (so the model/loss can ignore padded steps).
    """
    t, d = sequence.shape
    if t >= target_len:
        idx = np.linspace(0, t - 1, target_len).round().astype(int)
        fixed = sequence[idx]
        mask = np.ones(target_len, dtype=bool)
    else:
        fixed = np.zeros((target_len, d), dtype=sequence.dtype)
        fixed[:t] = sequence
        mask = np.zeros(target_len, dtype=bool)
        mask[:t] = True
    return fixed, mask
