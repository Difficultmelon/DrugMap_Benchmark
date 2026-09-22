# Data Card

## Intended Use

DragMapBench is intended for research on pharmaceutical and medical language
model evaluation, especially relation reasoning, pharmacokinetics,
repositioning, safety, ontology normalization, and robustness to presentation
changes.

## Composition

The V5.2 revised release contains 6,200 core questions and three diagnostic
families: 451 compositional bundles, 451 counterfactual triplets, and 2,877
answer-format groups. The revised subtype contains 451 items and spans 55
enzymes.

## Public/Private Boundary

Public files contain only model-facing inputs. Private files contain canonical
answers, accepted answers, scoring rules, and diagnostic labels. The private
files are intentionally absent from this repository to reduce test-set leakage.

## Revision

The active release is `revised_diverse_enzymes`. The revision removed generic
pronoun answers in the affected enzyme-exposure items and regenerated the
dependent diagnostic sets. The source revision digest is recorded in
`configs/dataset_manifest.json`.

## Limitations

Automatically generated diagnostic items should be treated as benchmark
diagnostics, not as expert-validated clinical guidance. Benchmark scores do
not constitute medical advice or evidence of clinical safety.
