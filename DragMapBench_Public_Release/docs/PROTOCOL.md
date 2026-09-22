# Formal Protocol

## Main Experiment

The main experiment evaluates 13 enabled benchmark models on the same 6,200
core questions. No external search or browsing is available. Each response is
stored as a structured answer and explanation, then scored against the private
gold rules.

## Experiment 1: Open Web

Experiment 1 repeats the 6,200 core questions with open-web access enabled.
The model may use the configured search tool, but the benchmark still requires
a structured answer and, where applicable, citations. Search settings are
declared in `configs/experiment.json`.

## Experiment 2: Compositional Diagnostics

Each of 451 bundles contains atomic prerequisite questions and a compositional
target. The scorer reports atomic prerequisite accuracy, target accuracy,
conditional reasoning accuracy, gold-atom utilization, and the
compositionality gap.

## Experiment 3: Counterfactual Diagnostics

Each of 451 triplets contains an original question, a lexical-control
question, and a causal-intervention question. The scorer reports original
accuracy, lexical invariance and instability, causal intervention accuracy,
causal sensitivity, and flip behavior.

## Experiment 4: Answer-Format Diagnostics

Each of 2,877 semantic groups contains one open-ended question and four
multiple-choice permutations. The scorer reports open-ended accuracy,
mean permutation accuracy, semantic consistency, all-permutations
consistency, option-position accuracy, and the recognition-recall gap.

## Scoring

Deterministic scoring follows the gold-declared method. It supports normalized
exact matching, Boolean matching, multiple-choice labels or text, numeric
precision intervals, and numeric interval-subset scoring. Semantic judging is
kept separate from deterministic scoring and is never silently substituted for
it.
