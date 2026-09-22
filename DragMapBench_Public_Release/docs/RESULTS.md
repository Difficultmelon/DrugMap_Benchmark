# Curated Results

The `results/summaries/` directory contains publication-facing summaries from
the latest completed runs in the private working project.

## Included Summaries

- `main_closed_book_13_models.json`: the 13-model closed-book baseline.
- `experiment1_open_web.json`: completed open-web core runs available in the
  latest working snapshot.
- `experiments2_4_diagnostics.json`: the latest complete diagnostic summaries
  for DeepSeek-V4.1-flash and the selected comparison models. Experiment 4
  now includes complete 2,877-group runs for GPT-5.6 Terra, Qwen-3.8-max,
  Gemma-4-E4B-it, Hulu-Med-7B, and DeepSeek-V4.1-flash.
- `release_result_manifest.json`: source run identifiers, completeness checks,
  and the no-raw-response publication policy.

The summaries contain metrics and sample counts only. They do not contain
private gold, raw model answers, provider request IDs, API metadata, SSH
details, or search traces.
