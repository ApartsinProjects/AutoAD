"""Per-source ranking and cross-source aggregation.

Each pseudo-anomaly source produces a per-candidate score per series.
We convert these scores to within-series ranks (1 = best detector) and
combine ranks across sources via Borda.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass
class SelectorOutput:
    """Per-series output of a selector."""

    series_id: str
    selector_id: str
    model_ids: list[str]
    per_model_rank: dict[str, float]   # 1.0 = best
    selected: str                       # argmin rank
    confidence: float                   # 0..1 (currently inter-source Kendall tau or NaN)


def scores_to_ranks(model_scores: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    """Convert raw per-model scores to ordinal ranks (1 = best)."""
    ids = list(model_scores)
    vals = np.array([model_scores[m] for m in ids], dtype=float)
    safe = np.where(np.isnan(vals), (-np.inf if higher_is_better else np.inf), vals)
    if higher_is_better:
        ranks = rankdata(-safe, method="average")
    else:
        ranks = rankdata(safe, method="average")
    return {m: float(r) for m, r in zip(ids, ranks)}


def borda_combine(source_ranks: list[dict[str, float]]) -> dict[str, float]:
    """Average per-model rank across sources; lower = better."""
    if not source_ranks:
        return {}
    ids = list(source_ranks[0])
    combined: dict[str, float] = {}
    for m in ids:
        vals = [src.get(m, np.nan) for src in source_ranks]
        vals = [v for v in vals if np.isfinite(v)]
        combined[m] = float(np.mean(vals)) if vals else float("nan")
    return combined


def kendall_agreement(ranks_a: dict[str, float], ranks_b: dict[str, float]) -> float:
    """Kendall tau-b between two rank dicts; NaN if degenerate."""
    from scipy.stats import kendalltau
    common = sorted(set(ranks_a) & set(ranks_b))
    if len(common) < 3:
        return float("nan")
    a = [ranks_a[m] for m in common]
    b = [ranks_b[m] for m in common]
    tau, _ = kendalltau(a, b)
    return float(tau) if tau is not None and np.isfinite(tau) else float("nan")


def select_from_ranks(
    series_id: str,
    selector_id: str,
    combined_ranks: dict[str, float],
    confidence: float = float("nan"),
) -> SelectorOutput:
    """Build a SelectorOutput by picking the model with the lowest rank."""
    if not combined_ranks:
        raise ValueError("Empty rank dict; cannot select")
    selected = min(combined_ranks, key=lambda m: combined_ranks[m])
    return SelectorOutput(
        series_id=series_id,
        selector_id=selector_id,
        model_ids=list(combined_ranks),
        per_model_rank=combined_ranks,
        selected=selected,
        confidence=confidence,
    )
