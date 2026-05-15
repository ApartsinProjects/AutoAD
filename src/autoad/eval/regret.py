"""Selection-regret computation.

For each series and each selector, compute::

    regret = Perf(oracle_best_model) - Perf(selector_pick)

Lower is better; 0 means the selector picked the oracle's best.
Aggregated across series via mean / median.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class RegretRecord:
    series_id: str
    selector_id: str
    selected_model: str
    selected_perf: float
    oracle_best_model: str
    oracle_best_perf: float
    regret: float


def compute_regret(
    series_id: str,
    selector_id: str,
    selected_model: str,
    oracle_perfs: dict[str, float],
) -> RegretRecord:
    """Compute regret for one (series, selector) pair.

    Parameters
    ----------
    oracle_perfs : dict[model_id -> performance]
        Oracle test-set performance for every candidate. Performance is
        an upward metric (higher = better), e.g. AUC-PR or VUS-PR.
    """
    finite = {m: v for m, v in oracle_perfs.items() if np.isfinite(v)}
    if not finite:
        raise ValueError(f"{series_id}: no finite oracle perfs")
    oracle_best_model = max(finite, key=lambda m: finite[m])
    oracle_best_perf = finite[oracle_best_model]
    selected_perf = oracle_perfs.get(selected_model, float("nan"))
    if not np.isfinite(selected_perf):
        # Selector picked a model the oracle couldn't evaluate; treat as worst-case regret.
        regret = oracle_best_perf
    else:
        regret = oracle_best_perf - selected_perf
    return RegretRecord(
        series_id=series_id,
        selector_id=selector_id,
        selected_model=selected_model,
        selected_perf=float(selected_perf) if np.isfinite(selected_perf) else float("nan"),
        oracle_best_model=oracle_best_model,
        oracle_best_perf=float(oracle_best_perf),
        regret=float(regret),
    )


def summarize_regret(records: Iterable[RegretRecord]) -> dict[str, dict[str, float]]:
    """Group records by selector_id and return basic statistics."""
    by_selector: dict[str, list[RegretRecord]] = {}
    for r in records:
        by_selector.setdefault(r.selector_id, []).append(r)
    out: dict[str, dict[str, float]] = {}
    for sel, rs in by_selector.items():
        regrets = np.array([r.regret for r in rs], dtype=float)
        finite = regrets[np.isfinite(regrets)]
        out[sel] = {
            "n": float(len(rs)),
            "mean_regret": float(np.mean(finite)) if len(finite) else float("nan"),
            "median_regret": float(np.median(finite)) if len(finite) else float("nan"),
            "min_regret": float(np.min(finite)) if len(finite) else float("nan"),
            "max_regret": float(np.max(finite)) if len(finite) else float("nan"),
            "top1_hit_rate": float(np.mean([r.selected_model == r.oracle_best_model for r in rs])),
        }
    return out
