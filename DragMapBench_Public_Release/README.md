# DragMapBench

DragMapBench is a formal pharmaceutical benchmark for drug relations,
pharmacokinetics, drug repositioning, safety, ontology normalization, and
clinical reasoning. This repository contains the latest V5.2
`revised_diverse_enzymes` release used for the paper experiments.

## Release Contents

- `data/public/`: model-facing questions and diagnostic inputs.
- `data/private/`: placeholder only; official gold files are distributed
  separately and are intentionally excluded from this public repository.
- `src/dragmap_formal/`: dataset validation, prompting, model clients,
  deterministic scoring, semantic judging, and diagnostic scorers.
- `scripts/run_formal_matrix.py`: unified runner for the main experiment and
  Experiments 1-4.
- `configs/`: public-safe, relative-path configurations with all models
  disabled by default.
- `results/summaries/`: curated metrics from the latest completed runs. Raw
  model responses, provider request IDs, and logs are not included.

## Dataset

The active release contains:

- 6,200 core questions and 6,200 private gold records.
- 451 compositional atomic-prerequisite bundles.
- 451 counterfactual triplets.
- 2,877 answer-format semantic groups.
- 451 revised `reduced_enzyme_activity_exposure` items covering 55 enzymes.

The public question files contain no answer, scoring, evidence, or verification
fields. The private gold files are required for official scoring and must not
be sent to evaluated models.

## Protocol

The paper reports five related layers:

1. **Main experiment:** 13 models answer all 6,200 questions closed-book.
2. **Experiment 1:** the same 6,200 questions with open-web access.
3. **Experiment 2:** atomic prerequisites are elicited before a compositional
   target is answered.
4. **Experiment 3:** original, lexical-control, and causal-intervention
   variants test invariance and causal sensitivity.
5. **Experiment 4:** an open-ended question is compared with four permuted
   multiple-choice presentations.

The formal layer definitions are documented in
[`docs/PROTOCOL.md`](docs/PROTOCOL.md).

## Installation

```bash
python -m venv .venv
# Activate the environment using the command for your shell.
python -m pip install -r requirements.txt
python -m pip install -e .
```

The public configuration disables all model entries. To run a local
reproduction, provide your own compatible endpoint and credentials through
`.env`; never commit `.env`.

## Data Audit

The public package can be inspected without private gold:

```bash
python scripts/audit_release.py
```

For official scoring, obtain the private gold package separately, place it
under `data/private/`, and then run the normal benchmark commands.

## Running the Matrix

After private gold and a compatible endpoint are configured:

```bash
python scripts/run_formal_matrix.py \
  --config configs/experiment.json \
  --models-config configs/models.json \
  --models "GPT-5.6 Terra" \
  --experiments main,experiment1,2,3,4
```

The unified runner fixes the main condition to `closed_book` and Experiment 1
to `open_web`. Experiments 2-4 are offline diagnostics.

## Results

See [`docs/RESULTS.md`](docs/RESULTS.md) for the release result policy and
links to curated summaries. The public repository does not redistribute raw
responses or private evaluation material.

## Citation and License

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The benchmark
code is released under the MIT License. Dataset and source-data attribution
must follow the notices in [`docs/DATA_CARD.md`](docs/DATA_CARD.md).
