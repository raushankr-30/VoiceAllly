"""STEP 20: Real latency measurement (spec section 21). Measures each
pipeline stage separately on the ACTUAL machine this is run on, and
records the hardware so numbers are interpretable, not just a bare
"real-time" claim.

Usage:
    python src/experiments/measure_latency.py --config experiments/E07_final/config.yaml \
        --checkpoint experiments/E07_final/checkpoint_best.pt --n-samples 50
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.features.dataset import LandmarkSequenceDataset, expected_feature_dim
from src.models.architectures import build_model
from src.utils.config import load_config


def hardware_info() -> dict:
    info = {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        info["ram_gb"] = "psutil not installed"
    return info


def percentile_or_nan(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if len(values) >= 5 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--out", default="results/tables/latency.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = Path(args.checkpoint).parent
    label_to_idx = json.loads((exp_dir / "label_to_idx.json").read_text())

    ds = LandmarkSequenceDataset(
        split_csv=cfg["test_csv"], landmarks_dir=cfg["landmarks_dir"], video_col=cfg["video_col"],
        label_col=cfg["label_col"], label_to_idx=label_to_idx, representation=cfg["representation"],
        feature_set=cfg["feature_set"], sequence_length=cfg["sequence_length"],
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    input_dim = expected_feature_dim(cfg["representation"], cfg["feature_set"])
    model = build_model(cfg["model"], input_dim=input_dim, num_classes=len(label_to_idx),
                         **cfg.get("model_kwargs", {})).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    inference_times = []
    with torch.no_grad():
        it = iter(loader)
        for _ in range(min(args.n_samples, len(ds))):
            batch = next(it)
            x = batch["features"].to(device)
            mask = batch["mask"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x, mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - t0)

    rows = [{
        "stage": "model_inference_per_sample",
        "mean_ms": float(np.mean(inference_times)) * 1000,
        "median_ms": float(np.median(inference_times)) * 1000,
        "p95_ms": percentile_or_nan(inference_times, 95) * 1000,
        "fps": 1.0 / float(np.mean(inference_times)) if np.mean(inference_times) > 0 else float("inf"),
        "n_samples": len(inference_times),
    }]

    print("[latency] NOTE: landmark-extraction and preprocessing latency (MediaPipe stage) "
          "must be measured separately by timing src/features/extract_landmarks.py's "
          "extract_one_video() per frame/video on the target hardware -- this script "
          "measures the model-inference stage only, on cached landmarks, which is what "
          "the E02-E04 configs actually vary. Combine both into total pipeline latency "
          "in results/paper_tables/ once both are measured on the same machine.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    hw_path = out_path.parent / "latency_hardware_info.json"
    hw_path.write_text(json.dumps(hardware_info(), indent=2))

    print(f"[latency] model_inference mean={rows[0]['mean_ms']:.2f}ms "
          f"fps={rows[0]['fps']:.1f} (n={rows[0]['n_samples']}) -> {out_path}")


if __name__ == "__main__":
    main()
