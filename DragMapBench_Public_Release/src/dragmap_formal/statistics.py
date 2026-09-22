from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable


def mean(values: Iterable[float]) -> float | None:
    values_list = list(values)
    return sum(values_list) / len(values_list) if values_list else None


def bootstrap_mean_ci(
    values: list[float],
    *,
    replicates: int = 10000,
    seed: int = 20260911,
    alpha: float = 0.05,
) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    n = len(values)
    samples: list[float] = []
    for _ in range(replicates):
        samples.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    low_idx = max(0, min(len(samples) - 1, int((alpha / 2) * len(samples))))
    high_idx = max(0, min(len(samples) - 1, int((1 - alpha / 2) * len(samples)) - 1))
    return {"mean": sum(values) / n, "ci_low": samples[low_idx], "ci_high": samples[high_idx]}


def paired_bootstrap_diff_ci(
    baseline: list[float],
    treatment: list[float],
    *,
    replicates: int = 10000,
    seed: int = 20260911,
    alpha: float = 0.05,
) -> dict[str, float | None]:
    if len(baseline) != len(treatment):
        raise ValueError("Paired bootstrap requires equally sized vectors")
    if not baseline:
        return {"diff": None, "ci_low": None, "ci_high": None}
    diffs = [b - a for a, b in zip(baseline, treatment)]
    ci = bootstrap_mean_ci(diffs, replicates=replicates, seed=seed, alpha=alpha)
    return {"diff": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}


def _comb(n: int, k: int) -> int:
    return math.comb(n, k)


def exact_mcnemar_pvalue(b_to_t: int, t_to_b: int) -> float | None:
    """Two-sided exact McNemar p-value using the binomial discordant count."""
    n = b_to_t + t_to_b
    if n == 0:
        return None
    k = min(b_to_t, t_to_b)
    cumulative = sum(_comb(n, i) * (0.5**n) for i in range(k + 1))
    return min(1.0, 2.0 * cumulative)


def paired_state_metrics(
    baseline_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
    *,
    id_key: str = "id",
    bootstrap_replicates: int = 10000,
    random_seed: int = 20260911,
) -> dict[str, Any]:
    baseline = {str(row[id_key]): bool(row.get("correct")) for row in baseline_rows if row.get(id_key) is not None}
    treatment = {str(row[id_key]): bool(row.get("correct")) for row in treatment_rows if row.get(id_key) is not None}
    common = sorted(set(baseline) & set(treatment))
    counts: dict[str, int] = defaultdict(int)
    for item_id in common:
        b = baseline[item_id]
        t = treatment[item_id]
        if b and t:
            counts["both_correct"] += 1
        elif not b and t:
            counts["rescued"] += 1
        elif b and not t:
            counts["harmed"] += 1
        else:
            counts["both_wrong"] += 1
    n = len(common)
    rescue = counts["rescued"]
    harm = counts["harmed"]
    baseline_wrong = rescue + counts["both_wrong"]
    baseline_correct = harm + counts["both_correct"]
    paired_bootstrap = paired_bootstrap_diff_ci(
        [float(baseline[item_id]) for item_id in common],
        [float(treatment[item_id]) for item_id in common],
        replicates=bootstrap_replicates,
        seed=random_seed,
    )
    return {
        "paired_items": n,
        "both_correct": counts["both_correct"],
        "rescued": rescue,
        "harmed": harm,
        "both_wrong": counts["both_wrong"],
        "closed_book_accuracy": baseline_correct / n if n else None,
        "open_web_accuracy": (counts["both_correct"] + rescue) / n if n else None,
        "accuracy_gain": (rescue - harm) / n if n else None,
        "web_rescue_rate": rescue / baseline_wrong if baseline_wrong else None,
        "web_harm_rate": harm / baseline_correct if baseline_correct else None,
        # Retained as aliases for compatibility with prior reports.
        "rescue_rate": rescue / baseline_wrong if baseline_wrong else None,
        "harm_rate": harm / baseline_correct if baseline_correct else None,
        "net_gain": (rescue - harm) / n if n else None,
        "mcnemar_exact_p": exact_mcnemar_pvalue(rescue, harm),
        "paired_bootstrap_accuracy_gain": paired_bootstrap,
    }


def proportion(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
