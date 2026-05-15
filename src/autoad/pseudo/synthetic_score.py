"""Source 2: synthetic-perturbation pseudo-anomalies.

For each candidate detector, score its ability to detect synthetic
anomalies injected into held-out normal data. Returns one AUC-PR per
``(candidate, family)`` pair, which the caller aggregates across
families to produce a per-candidate Source-2 score.

The injected anomalies are deliberately disjoint from the real test-set
anomalies the oracle uses; this remains zero-label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.synthetic import inject_contextual, INJECTORS
from ..data.windowing import aggregate_window_labels, scores_to_per_point, sliding_windows
from ..eval.metrics import auc_pr
from ..models.base import BaseAD


@dataclass
class SyntheticScoreResult:
    """Per-series Source-2 record."""

    series_id: str
    model_ids: list[str]
    # per-family AUCs: {family -> {model_id -> auc_pr}}
    per_family_aucs: dict[str, dict[str, float]]
    # per-candidate score: mean AUC-PR across families
    per_model_score: dict[str, float]


def run_synthetic_source(
    *,
    series_id: str,
    fitted_models: list[tuple[str, BaseAD]],
    normal_signal: np.ndarray,
    window: int = 64,
    stride: int = 1,
    families: tuple[str, ...] = ("point_spike", "level_shift", "trend_change", "frequency_change"),
    seed: int = 42,
) -> SyntheticScoreResult:
    """Score each fitted model on synthetic-perturbation pseudo-anomalies.

    Parameters
    ----------
    fitted_models : list of (model_id, BaseAD)
        Models already fit on the series' normal training data. They are
        scored, not re-fit.
    normal_signal : 1-D array
        Clean normal data used to construct perturbed test signals.
        Should be disjoint from any real-anomaly evaluation set.
    """
    model_ids = [mid for mid, _ in fitted_models]
    per_family_aucs: dict[str, dict[str, float]] = {}

    rng = np.random.default_rng(seed)
    for family in families:
        # Build perturbed test signal from a slice of the normal signal
        seed_f = int(rng.integers(0, 10**8))
        if family == "contextual":
            # Choose a period; default 50 since synthetic_v1 uses it
            x_pert, labels = inject_contextual(normal_signal.copy(), period=50, seed=seed_f)
        elif family in INJECTORS:
            x_pert, labels = INJECTORS[family](normal_signal.copy(), seed=seed_f)
        else:
            continue
        if labels.sum() == 0:
            continue
        # Window the perturbed signal and the labels
        if len(x_pert) <= window:
            continue
        wins = sliding_windows(x_pert, window, stride)
        win_labels = aggregate_window_labels(labels, window, stride, mode="max")
        # Score each model and record AUC-PR
        family_aucs: dict[str, float] = {}
        for model_id, model in fitted_models:
            try:
                scores = model.score(wins)
                pp = scores_to_per_point(scores, len(x_pert), window, stride, reducer="max")
                family_aucs[model_id] = float(auc_pr(labels, pp))
            except Exception:
                family_aucs[model_id] = float("nan")
        per_family_aucs[family] = family_aucs

    # Aggregate per-candidate score: mean AUC-PR across families
    per_model_score: dict[str, float] = {}
    for mid in model_ids:
        vals = [per_family_aucs[f][mid] for f in per_family_aucs if mid in per_family_aucs[f]]
        vals = [v for v in vals if np.isfinite(v)]
        per_model_score[mid] = float(np.mean(vals)) if vals else float("nan")

    return SyntheticScoreResult(
        series_id=series_id,
        model_ids=model_ids,
        per_family_aucs=per_family_aucs,
        per_model_score=per_model_score,
    )
