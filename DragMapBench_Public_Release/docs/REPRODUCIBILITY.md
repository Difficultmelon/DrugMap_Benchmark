# Reproducibility

## Public Check

From the repository root:

```bash
python scripts/audit_release.py
python -m pytest -q
python -m compileall -q src scripts tests
```

The public check validates the release structure without requiring private gold
or an API endpoint.

## Official Evaluation

1. Obtain the private core and diagnostic gold package through the project
   release channel.
2. Place the files under `data/private/` using the paths in
   `configs/experiment.json`.
3. Copy `.env.example` to `.env` and fill in credentials locally.
4. Enable only the model entries and endpoints intended for your run.
5. Run a small `--limit` smoke test before the full matrix.
6. Keep `runs/`, `reports/`, `.env`, and raw provider responses outside the
   public release.

The public release does not prescribe a provider. Any OpenAI-compatible
endpoint must return a structured JSON answer with the fields required by the
benchmark prompts.
