# VoiceAlly -- ISL Recognition Research Pipeline

Lightweight, landmark-based Indian Sign Language recognition on CISLR,
evaluated for recognition quality, signer-independent generalization,
robustness, and real-time efficiency. This repo is the experimental
core for the IEEE paper -- not a product.

**Current status:** scaffolding complete, unexecuted (no dataset in
`data/raw/` yet). See `research/IMPLEMENTATION_STATUS.md` for the exact
state.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline (run in this order once `data/raw/` has real CISLR data)

```bash
# 1. Schema-agnostic audit -- writes research/RAW_SCHEMA_REPORT.md
python src/data/audit_dataset.py --raw-dir data/raw

# 2. Fill in configs/column_mapping.yaml using that report, then:
python src/data/audit_dataset.py --raw-dir data/raw --mapping configs/column_mapping.yaml

# 3. Vocabulary selection
python src/data/select_vocabulary.py --mapping configs/column_mapping.yaml

# 4. Signer-independent split (fails loudly if signer metadata isn't real)
python src/data/split_data.py --mapping configs/column_mapping.yaml

# 5. Verify no leakage
python src/data/check_leakage.py --mapping configs/column_mapping.yaml

# 6. Extract + cache landmarks (run once; cached afterwards)
python src/features/extract_landmarks.py --mapping configs/column_mapping.yaml

# 7. Determine sequence_length from the real frame-count distribution,
#    then fill it into configs/*.yaml (see src/features/temporal.py)

# 8. Experiment 1: landmark modality (R1 vs R2 vs R3)
python src/experiments/run_experiment.py --configs-dir configs/E02_modality \
    --exp-root experiments/E02_modality --out results/tables/modality_comparison.csv

# 9. Experiment 2: temporal model (MLP vs LSTM vs GRU vs TCN), after
#    updating configs/E03_temporal/*.yaml's `representation` to the
#    Experiment 1 winner
python src/experiments/run_experiment.py --configs-dir configs/E03_temporal \
    --exp-root experiments/E03_temporal --out results/tables/model_comparison.csv

# 10. Experiment 3: feature ablation (F1/F2/F3), after updating
#     configs/E04_features/*.yaml similarly
python src/experiments/run_experiment.py --configs-dir configs/E04_features \
    --exp-root experiments/E04_features --out results/tables/feature_ablation.csv

# 11. Freeze the winning config into research/FINAL_CONFIGURATION.md,
#     train it once more as the canonical final run, then:
python src/training/evaluate.py --config experiments/E07_final/config.yaml \
    --checkpoint experiments/E07_final/checkpoint_best.pt --split test

# 12. Latency, robustness, error analysis
python src/experiments/measure_latency.py --config experiments/E07_final/config.yaml \
    --checkpoint experiments/E07_final/checkpoint_best.pt
python src/experiments/robustness.py --config experiments/E07_final/config.yaml \
    --checkpoint experiments/E07_final/checkpoint_best.pt
python src/experiments/error_analysis.py --config experiments/E07_final/config.yaml \
    --checkpoint experiments/E07_final/checkpoint_best.pt

# 13. Live webcam validation (on a machine with a camera)
python src/live/webcam_demo.py --config experiments/E07_final/config.yaml \
    --checkpoint experiments/E07_final/checkpoint_best.pt --signer-name <name>

# 14. Paper-ready table export
python src/experiments/generate_paper_tables.py
```

## Repository layout
```
research/            Dataset card, implementation status, final config, findings
src/data/            Audit, vocabulary selection, splitting, leakage check
src/features/        Landmark spec (R1/R2/R3), normalization, temporal
                      sampling, kinematic (F1/F2/F3) features, extraction,
                      training-time dataset assembly
src/models/           Baseline MLP, LSTM, GRU, TCN
src/training/         train.py (val-only model selection), evaluate.py (test, once)
src/experiments/       latency, robustness, error analysis, orchestration, paper tables
src/live/              Live webcam validation
src/language/          Minimal recognized-sign -> text (+ optional TTS)
configs/                Column mapping + per-experiment YAML configs
data/, results/, experiments/  Populated by running the pipeline (currently empty)
```

## Ground rules this code follows
- No fabricated numbers, ever. Anything not measured is labeled
  `NOT YET MEASURED` (see `research/FINDINGS.md`).
- No assumed CISLR column names (see `src/data/audit_dataset.py`).
- Test set is touched exactly once, after the config is frozen
  (`src/training/evaluate.py`).
- Signer-independent split is refused, not faked, if signer metadata
  isn't real (`src/data/split_data.py`).
