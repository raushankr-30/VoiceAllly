# Implementation Status

## What already existed in the repository
The supplied scaffold contains pipeline source in `src/`, configuration
templates in `configs/`, and documentation templates in `research/`. It
also contains 1,500 extracted MP4 files in `data/raw/Data_1500/` and two
video-only ZIP archives.

## What works right now
The annotations are now present and audited. `dataset.csv` supplies
video-to-gloss labels; the local manifest covers every one of the 7,050 MP4
files. PyYAML was installed to execute audit and vocabulary scripts. OpenCV,
MediaPipe, and PyTorch remain unavailable.

Everything below is **scaffolded and believed correct against the spec**,
but **unexecuted** until real data lands in `data/raw/`.

## What is missing (blocking real execution)
1. A larger labeled sample per gloss. This delivery has 1--13 videos per
   gloss, so no class meets the fixed 15-video rule.
2. Reliable signer IDs for signer-independent evaluation.
2. `configs/column_mapping.yaml` filled in against the *real* CISLR schema
   (label column, video path column, signer column if any, duration/fps
   if present) -- deliberately left as `null` placeholders; see
   `src/data/audit_dataset.py`.
3. A real `sequence_length` chosen from the real frame-count distribution
   (currently `null` in every config).
4. Everything downstream of that: vocabulary selection, splits, leakage
   check, landmark extraction, all four experiments, final model
   selection, test evaluation, latency, robustness, error analysis, live
   validation.

## What was implemented in this session
Full pipeline code, matching the execution order in spec section 35,
STEP 1-2 and STEP 3-25 as far as they can be written without touching
real data:

| Area | Files |
|---|---|
| Config / seed / metrics utilities | `src/utils/` |
| Dataset audit (schema-agnostic) | `src/data/audit_dataset.py` |
| Vocabulary selection rule | `src/data/select_vocabulary.py` |
| Signer-independent (group-aware) split | `src/data/split_data.py` |
| Leakage PASS/FAIL check | `src/data/check_leakage.py` |
| Landmark representation spec (R1/R2/R3) | `src/features/landmark_spec.py` |
| Normalization (spec section 10) | `src/features/normalize.py` |
| Temporal sampling/padding (spec section 11) | `src/features/temporal.py` |
| Feature ablation: position/velocity/acceleration | `src/features/kinematics.py` |
| MediaPipe Holistic extraction + caching | `src/features/extract_landmarks.py` |
| Training-time dataset assembly (ties repr/norm/temporal/features together) | `src/features/dataset.py` |
| Models: baseline MLP, LSTM, GRU, TCN | `src/models/architectures.py` |
| Train / validate loop | `src/training/train.py` |
| Frozen-model test evaluation | `src/training/evaluate.py` |
| Latency measurement + hardware logging | `src/experiments/measure_latency.py` |
| Robustness (3 of 5 perturbations implementable at landmark level; 2 explicitly marked NOT YET MEASURED) | `src/experiments/robustness.py` |
| Error analysis (confusion pairs + sample lookup, no invented causes) | `src/experiments/error_analysis.py` |
| Experiment orchestration (E02/E03/E04 sweep runner) | `src/experiments/run_experiment.py` |
| Paper table export | `src/experiments/generate_paper_tables.py` |
| Live webcam validation | `src/live/webcam_demo.py` |
| Minimal recognized-sign -> text (+ optional TTS) demo | `src/language/translate.py` |
| Config templates for E02 (R1/R2/R3), E03 (MLP/LSTM/GRU/TCN), E04 (F1/F2/F3) | `configs/` |

## Deliberate honesty gaps (not bugs)
- `configs/column_mapping.yaml` and every `sequence_length` field are
  `null` on purpose -- filling them requires the real CISLR schema and
  real frame-count distribution, which do not exist in this container yet.
- `src/data/split_data.py` **refuses** to produce a signer-independent
  split (and will not silently fall back) unless a real, reliable signer
  column is confirmed present -- this is enforced in code, not just
  documentation.
- `src/experiments/robustness.py` explicitly reports `lighting_variation`
  and `background_variation` as `NOT YET MEASURED`, because those require
  video-level re-augmentation + re-extraction, not a landmark-level proxy.
- No `research/DATASET_CARD.md`, `research/FINAL_CONFIGURATION.md`, or
  `research/FINDINGS.md` contain real numbers yet -- they are templates
  with `NOT YET MEASURED` placeholders, per spec section 30.

## What I will implement/run next, in order (spec section 35)
1. `python src/data/audit_dataset.py --raw-dir data/raw` once CISLR is
   uploaded, to get the real schema report.
2. Fill in `configs/column_mapping.yaml` from that report.
3. Re-run the audit with `--mapping` to get real dataset statistics and
   figures, and determine whether signer metadata genuinely exists.
4. Proceed through vocabulary selection, splitting, leakage check,
   extraction, and the three experiments, in the order in spec section 35.
