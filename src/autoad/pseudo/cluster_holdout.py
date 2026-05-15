"""Leave-Cluster-Out (LCO) validation, the central novel component of MS-PAS.

Algorithm (Section 5.1 of the implementation plan)
--------------------------------------------------
Given a single time series and a candidate detector pool:

1. Extract feature vectors of training windows. Two variants:
   - "data" : :func:`autoad.data.features.summary_features`
   - "latent": :class:`autoad.data.features.PCAEncoder`
2. Cluster the feature vectors with each ``(cluster_algo, C)`` config.
3. For each cluster :math:`C_j` passing fairness filters (size and
   difficulty), define a fold:
   - train each candidate on windows from :math:`X_N \\setminus C_j`,
   - score :math:`C_j` (pseudo-anomalous) and an IID slice of
     :math:`X_N \\setminus C_j` (pseudo-normal),
   - compute AUC-PR on the pseudo labels.
4. Aggregate per-candidate AUCs across folds, weighted equally
   per difficulty bucket so that no single bucket dominates.
5. Convert per-(algo, C) AUCs to ranks and Borda-aggregate across all
   ``(algo, C)`` configurations to produce a single per-candidate
   LCO score per series.

Returns a dict that can be persisted to parquet by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from ..eval.metrics import auc_pr
from ..models.base import BaseAD
from .difficulty import VALID_BUCKETS, difficulty_bucket, logreg_separability


# ----------------------------------------------------------------------
# Cluster algorithm registry
# ----------------------------------------------------------------------

def fit_clusters(
    features: np.ndarray,
    algo: str,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """Return integer cluster labels of shape ``(n_windows,)``."""
    if algo == "kmeans":
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        return km.fit_predict(features)
    if algo == "gmm":
        gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        return gmm.fit_predict(features)
    raise ValueError(f"Unknown cluster algorithm {algo!r}")


# ----------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------

@dataclass
class FoldResult:
    """Per-fold record. One row per (cluster_algo, C, cluster_idx)."""

    algo: str
    C: int
    cluster_idx: int
    n_in_cluster: int
    n_train_used: int
    difficulty_auc: float
    bucket: str
    model_aucs: dict[str, float] = field(default_factory=dict)  # per model_id


@dataclass
class LCOResult:
    """Per-series LCO summary: per-model rank across folds + fold details."""

    series_id: str
    variant: str  # "data" or "latent"
    model_ids: list[str]
    n_folds_used: int
    n_folds_excluded: int
    per_model_score: dict[str, float]   # mean rank across valid folds (lower = better)
    per_model_rank: dict[str, int]      # ordinal rank from per_model_score
    folds: list[FoldResult] = field(default_factory=list)


# ----------------------------------------------------------------------
# Core LCO routine
# ----------------------------------------------------------------------

def run_lco_one_series(
    *,
    series_id: str,
    train_windows: np.ndarray,
    features: np.ndarray,
    candidate_pool: list[tuple[str, BaseAD]],
    cluster_configs: list[tuple[str, int]],
    variant: str,
    min_cluster_size: int = 30,
    valid_buckets: tuple[str, ...] = VALID_BUCKETS,
    rng_seed: int = 42,
) -> LCOResult:
    """Run LCO on one series for a fixed candidate pool.

    Parameters
    ----------
    series_id : str
        Series identifier (for the result record).
    train_windows : (n_train, w) array
        Raw training windows (used to fit candidate detectors).
    features : (n_train, d_feat) array
        Feature vectors aligned 1:1 with ``train_windows``. The clustering
        operates on this matrix; the detectors are fit on ``train_windows``.
    candidate_pool : list of ``(model_id, BaseAD)``
        Each ``BaseAD`` is an unfitted candidate instance.
    cluster_configs : list of ``(algo, C)``
        Cluster algorithm and granularity pairs to try.
    variant : str
        "data" or "latent", recorded on the result for downstream merge.
    min_cluster_size : int
        Drop folds whose held-out cluster has fewer windows than this.
    rng_seed : int
        Seed for the in-train pseudo-normal slice sampling.
    """
    rng = np.random.default_rng(rng_seed)
    model_ids = [mid for mid, _ in candidate_pool]
    n_train = train_windows.shape[0]

    folds: list[FoldResult] = []
    for algo, C in cluster_configs:
        if C >= n_train // 2:
            continue
        try:
            labels = fit_clusters(features, algo, C, random_state=rng_seed)
        except Exception:
            continue

        for cluster_idx in range(C):
            in_mask = labels == cluster_idx
            n_in = int(in_mask.sum())
            if n_in < min_cluster_size:
                continue
            train_mask = ~in_mask
            n_train_used = int(train_mask.sum())
            if n_train_used < min_cluster_size:
                continue

            train_feats = features[train_mask]
            holdout_feats = features[in_mask]

            diff_auc = logreg_separability(train_feats, holdout_feats)
            bucket = difficulty_bucket(diff_auc)
            if bucket not in valid_buckets:
                # Trivial / impossible / invalid: still record but model_aucs empty
                folds.append(
                    FoldResult(
                        algo=algo, C=C, cluster_idx=cluster_idx,
                        n_in_cluster=n_in, n_train_used=n_train_used,
                        difficulty_auc=diff_auc, bucket=bucket,
                    )
                )
                continue

            # Build pseudo-normal sample (random slice of training cluster windows)
            n_eval_normal = min(n_in, n_train_used)
            train_indices = np.where(train_mask)[0]
            pseudo_normal_idx = rng.choice(train_indices, size=n_eval_normal, replace=False)
            pseudo_anom_idx = np.where(in_mask)[0]

            eval_idx = np.concatenate([pseudo_normal_idx, pseudo_anom_idx])
            eval_labels = np.concatenate(
                [np.zeros(len(pseudo_normal_idx), dtype=np.uint8),
                 np.ones(len(pseudo_anom_idx), dtype=np.uint8)]
            )

            model_aucs: dict[str, float] = {}
            for model_id, model in candidate_pool:
                try:
                    fitted = type(model)(**model.hyperparams).fit(train_windows[train_mask])
                    scores = fitted.score(train_windows[eval_idx])
                    model_aucs[model_id] = float(auc_pr(eval_labels, scores))
                except Exception:
                    model_aucs[model_id] = float("nan")

            folds.append(
                FoldResult(
                    algo=algo, C=C, cluster_idx=cluster_idx,
                    n_in_cluster=n_in, n_train_used=n_train_used,
                    difficulty_auc=diff_auc, bucket=bucket,
                    model_aucs=model_aucs,
                )
            )

    # ---------------------------- aggregate ----------------------------
    valid_folds = [f for f in folds if f.bucket in valid_buckets and f.model_aucs]
    excluded = len(folds) - len(valid_folds)

    # Difficulty-stratified mean rank per model
    per_model_score = _aggregate_ranks(model_ids, valid_folds, valid_buckets=valid_buckets)
    # Ordinal rank from those scores (lower rank-mean = better detector).
    # NaN scores -> infinity so the model gets the worst (largest) rank.
    score_arr = np.array([per_model_score[m] for m in model_ids], dtype=float)
    safe = np.where(np.isnan(score_arr), np.inf, score_arr)
    order = rankdata(safe, method="average")
    per_model_rank = {m: int(round(float(r))) for m, r in zip(model_ids, order)}

    return LCOResult(
        series_id=series_id, variant=variant, model_ids=model_ids,
        n_folds_used=len(valid_folds), n_folds_excluded=excluded,
        per_model_score=per_model_score, per_model_rank=per_model_rank,
        folds=folds,
    )


def _aggregate_ranks(
    model_ids: list[str],
    valid_folds: list[FoldResult],
    valid_buckets: tuple[str, ...] = VALID_BUCKETS,
) -> dict[str, float]:
    """Difficulty-stratified mean rank per model.

    Within each (bucket) we average per-model ranks across folds; then
    we average those bucket means with equal weight per bucket so that
    no single bucket dominates.
    """
    if not valid_folds:
        return {m: float("nan") for m in model_ids}
    by_bucket: dict[str, list[dict[str, float]]] = {b: [] for b in valid_buckets}
    for f in valid_folds:
        ranked = _rank_within_fold(model_ids, f.model_aucs)
        by_bucket.setdefault(f.bucket, []).append(ranked)
    bucket_means: list[dict[str, float]] = []
    for bucket, fold_ranks in by_bucket.items():
        if not fold_ranks:
            continue
        means = {m: float(np.nanmean([fr[m] for fr in fold_ranks])) for m in model_ids}
        bucket_means.append(means)
    if not bucket_means:
        return {m: float("nan") for m in model_ids}
    out = {m: float(np.nanmean([bm[m] for bm in bucket_means])) for m in model_ids}
    return out


def _rank_within_fold(model_ids: list[str], fold_aucs: dict[str, float]) -> dict[str, float]:
    """Higher AUC = lower rank number (rank 1 = best). NaNs get worst rank."""
    vals = np.array([fold_aucs.get(m, np.nan) for m in model_ids], dtype=float)
    # rankdata with 'min' gives ties the lower rank; we negate so that
    # higher AUC produces lower rank (= better).
    safe = np.where(np.isnan(vals), -np.inf, vals)
    ranks = rankdata(-safe, method="average")
    return {m: float(r) for m, r in zip(model_ids, ranks)}


# ----------------------------------------------------------------------
# Cross-config Borda aggregation (helper for callers)
# ----------------------------------------------------------------------

def borda_across_results(
    results: list[LCOResult], model_ids: list[str]
) -> dict[str, float]:
    """Mean per-model score (rank mean) across multiple LCOResults.

    Useful for combining data-domain and latent-domain LCO runs.
    """
    scores = {m: [] for m in model_ids}
    for r in results:
        for m in model_ids:
            v = r.per_model_score.get(m, np.nan)
            if np.isfinite(v):
                scores[m].append(v)
    return {m: float(np.mean(vs)) if vs else float("nan") for m, vs in scores.items()}
