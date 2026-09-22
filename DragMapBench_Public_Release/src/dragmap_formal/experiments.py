"""Public API for the main experiment and four formal DragMapBench experiments."""

from .core import compare_core_runs, run_core, score_core
from .diagnostics import (
    run_answer_format,
    run_compositional,
    run_counterfactual,
    score_answer_format,
    score_compositional,
    score_counterfactual,
)

EXPERIMENTS = {
    "main": "closed_book_core",
    "experiment1": "open_web_core",
    "experiment2": "compositional",
    "experiment3": "counterfactual",
    "experiment4": "answer_format",
}

__all__ = [
    "EXPERIMENTS",
    "compare_core_runs",
    "run_core",
    "score_core",
    "run_compositional",
    "score_compositional",
    "run_counterfactual",
    "score_counterfactual",
    "run_answer_format",
    "score_answer_format",
]
