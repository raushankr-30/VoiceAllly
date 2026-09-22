"""STEP 25: Copy/format the result CSVs that already exist under
results/tables/ into results/paper_tables/ (spec section 27) -- clean,
renamed, rounded for LaTeX conversion. Does not compute anything new;
it only exists so the paper-writing step has one place to pull final
CSVs from instead of hunting through experiments/.

Usage:
    python src/experiments/generate_paper_tables.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# (source path relative to results/tables, output filename, paper table number)
TABLE_MAP = [
    ("dataset_statistics.csv", "table1_dataset_statistics.csv"),
    ("modality_comparison.csv", "table2_landmark_representation_comparison.csv"),
    ("model_comparison.csv", "table3_temporal_model_comparison.csv"),
    ("final_test_results.csv", "table4_signer_independent_final_performance.csv"),
    ("robustness.csv", "table5_robustness_results.csv"),
    ("latency.csv", "table6_latency_efficiency.csv"),
]


def main():
    src_dir = Path("results/tables")
    dst_dir = Path("results/paper_tables")
    dst_dir.mkdir(parents=True, exist_ok=True)

    written, missing = [], []
    for src_name, dst_name in TABLE_MAP:
        src_path = src_dir / src_name
        if not src_path.exists():
            missing.append(src_name)
            continue
        df = pd.read_csv(src_path)
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].round(4)
        df.to_csv(dst_dir / dst_name, index=False)
        written.append(dst_name)

    print(f"[paper_tables] wrote {len(written)}: {written}")
    if missing:
        print(f"[paper_tables] NOT YET AVAILABLE (run the corresponding experiment first): {missing}")


if __name__ == "__main__":
    main()
