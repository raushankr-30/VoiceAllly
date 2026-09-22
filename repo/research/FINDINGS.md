# Research Findings

**Status: LABELS AUDITED; CORE SUPERVISED EXPERIMENT BLOCKED BY SAMPLE DENSITY.** Per
project rules (spec section 30/38), no number here may be filled in
until the corresponding experiment has actually been run against real
CISLR data, and no conclusion may be written until the evidence exists.

## Dataset
7,050 MP4 files have official gloss annotations. The corpus contains 4,764
glosses with 1--13 examples each (median 1); no signer metadata exists.

## Vocabulary
0 of 4,764 glosses satisfy the fixed minimum of 15 usable videos. See
`data/processed/vocabulary.csv`.

## Evaluation Protocol
Not run. Signer IDs are absent, and the complete corpus contains no class with
enough examples for a credible supervised train/validation/test split.

## Landmark Representation
NOT YET MEASURED -- see `results/tables/modality_comparison.csv`.

## Temporal Model
NOT YET MEASURED -- see `results/tables/model_comparison.csv`.

## Signer Independence
Not possible with the supplied metadata because no signer ID is present.

## Robustness
NOT YET MEASURED -- see `results/tables/robustness.csv`. Note: lighting
and background perturbations require video-level augmentation and are
separately flagged as not-yet-implemented even once landmark-level
perturbations are run.

## Efficiency
NOT YET MEASURED -- see `results/tables/latency.csv`.

## Error Analysis
NOT YET INSPECTED -- see `results/tables/error_analysis.csv`.

## Live Validation
NOT YET RUN -- see `results/live_test.csv`.

## Main Research Finding
NOT YET AVAILABLE. Will be written as 2-4 precise, evidence-backed
conclusions once the above sections are populated -- not before.

## Limitations
The official labels were supplied, but the full 7,050-video corpus is still
too sparse for the requested supervised experiments. A dataset with repeated
samples per gloss and reliable signer IDs is required.
