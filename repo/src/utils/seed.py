"""Deterministic seeding shared by every script in the pipeline.

Called at the top of every entrypoint (audit, split, extraction, training,
evaluation) so that a given config + seed is fully reproducible, per
research/FINAL_CONFIGURATION.md once it exists.
"""
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms where available. Some ops (e.g. certain
        # RNN backward passes on GPU) do not have a deterministic kernel;
        # we do not force-crash on that, we just record best-effort determinism.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
