# Final Frozen Configuration

**Status: NOT YET SELECTED.** This file must only be written once, after
Experiments 1-3 (landmark modality, temporal model, feature ablation)
have been run and a final configuration has been chosen using
VALIDATION metrics only (spec section 17). After this file is written,
the model/representation/feature-set/hyperparameters below must not
change based on test-set results.

## Selected representation
NOT YET SELECTED (winner of `results/tables/modality_comparison.csv` on
validation macro-F1, also weighed against latency/complexity per spec
section 17).

## Selected temporal model
NOT YET SELECTED (winner of `results/tables/model_comparison.csv`).

## Selected feature set
NOT YET SELECTED (winner of the F1/F2/F3 ablation, spec section 16 --
skipped/reported as such if not computationally feasible).

## Exact frozen config
NOT YET FROZEN. Once selected, copy the exact winning YAML content here
verbatim (not just a pointer to the file, so this document survives even
if `experiments/` is later cleaned up), including: seed, sequence_length,
batch_size, learning_rate, epochs, early_stopping_patience,
model_kwargs.

## Checkpoint location
NOT YET TRAINED -> will point to `experiments/E07_final/checkpoint_best.pt`.

## Rationale
NOT YET WRITTEN. Must state, in 2-3 sentences, why this configuration
was chosen over the alternatives -- referencing actual validation
numbers, not general reasoning.
