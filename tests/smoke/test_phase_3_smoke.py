"""Phase-3 smoke test: LCO + Source 2 + aggregation + regret on 3 synthetic series.

Tiny scope to catch regressions in seconds without external data.
"""

from __future__ import annotations

import numpy as np
import pytest

from autoad.data.features import PCAEncoder, summary_features
from autoad.data.synthetic_suite import build_suite
from autoad.data.windowing import sliding_windows
from autoad.eval.regret import compute_regret, summarize_regret
from autoad.models.classical import IForest, LOF
from autoad.pseudo.cluster_holdout import run_lco_one_series
from autoad.pseudo.difficulty import difficulty_bucket, logreg_separability
from autoad.pseudo.synthetic_score import run_synthetic_source
from autoad.selection.aggregate import (
    borda_combine,
    kendall_agreement,
    scores_to_ranks,
    select_from_ranks,
)


WINDOW = 32


def _candidates() -> list[tuple[str, object]]:
    return [
        ("iforest_50", IForest(n_estimators=50, random_state=42)),
        ("iforest_200", IForest(n_estimators=200, random_state=42)),
        ("lof_10", LOF(n_neighbors=10)),
    ]


@pytest.mark.smoke
def test_summary_features_panel_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(50, WINDOW)).astype(np.float32)
    F = summary_features(X)
    assert F.shape == (50, 15)
    assert np.isfinite(F).all()


@pytest.mark.smoke
def test_pca_encoder_dimension_reduction():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(50, WINDOW)).astype(np.float32)
    enc = PCAEncoder(n_components=8).fit(X)
    Z = enc.transform(X)
    assert Z.shape == (50, 8)
    assert np.isfinite(Z).all()


@pytest.mark.smoke
def test_difficulty_diagnostic_monotone():
    """Well-separated features should produce AUC near 1; overlapping near 0.5."""
    rng = np.random.default_rng(0)
    n = 60
    A = rng.normal(0, 1, size=(n, 5))
    B_sep = rng.normal(5, 1, size=(n, 5))  # clearly separable
    B_overlap = rng.normal(0, 1, size=(n, 5))  # same distribution
    auc_sep = logreg_separability(A, B_sep)
    auc_overlap = logreg_separability(A, B_overlap)
    assert auc_sep > 0.9, f"separable case AUC={auc_sep}"
    assert 0.3 <= auc_overlap <= 0.7, f"overlap case AUC={auc_overlap}"
    assert difficulty_bucket(0.99) == "trivial"
    assert difficulty_bucket(0.45) == "impossible"
    assert difficulty_bucket(0.80) == "medium"


@pytest.mark.smoke
def test_lco_data_domain_on_synthetic_series():
    """LCO produces a complete per-model ranking on a synthetic series."""
    series = build_suite(["level_shift"], n_per_family=1, length=2000)[0]
    train_w = sliding_windows(series.x_train, WINDOW, 1)
    features = summary_features(train_w)
    candidates = _candidates()
    result = run_lco_one_series(
        series_id=series.series_id,
        train_windows=train_w,
        features=features,
        candidate_pool=candidates,
        cluster_configs=[("kmeans", 4)],
        variant="data",
        valid_buckets=("trivial", "easy", "medium", "hard"),
    )
    assert result.n_folds_used >= 1, f"no valid folds, excluded={result.n_folds_excluded}"
    assert set(result.per_model_score) == {mid for mid, _ in candidates}
    assert set(result.per_model_rank) == {mid for mid, _ in candidates}
    # All scores finite
    for v in result.per_model_score.values():
        assert np.isfinite(v), f"non-finite score in {result.per_model_score}"


@pytest.mark.smoke
def test_lco_latent_variant_works():
    """LCO with PCA-latent features runs end-to-end."""
    series = build_suite(["trend_change"], n_per_family=1, length=2000)[0]
    train_w = sliding_windows(series.x_train, WINDOW, 1)
    enc = PCAEncoder(n_components=8).fit(train_w)
    features = enc.transform(train_w)
    candidates = _candidates()
    result = run_lco_one_series(
        series_id=series.series_id,
        train_windows=train_w,
        features=features,
        candidate_pool=candidates,
        cluster_configs=[("kmeans", 4)],
        variant="latent",
        valid_buckets=("trivial", "easy", "medium", "hard"),
    )
    assert result.variant == "latent"
    assert result.n_folds_used >= 1


@pytest.mark.smoke
def test_synthetic_source_returns_per_family_aucs():
    series = build_suite(["frequency_change"], n_per_family=1, length=2000)[0]
    train_w = sliding_windows(series.x_train, WINDOW, 1)
    fitted = [(mid, model.fit(train_w)) for mid, model in _candidates()]
    s2 = run_synthetic_source(
        series_id=series.series_id,
        fitted_models=fitted,
        normal_signal=series.x_train.astype(np.float32),
        window=WINDOW,
        families=("point_spike", "level_shift"),
    )
    assert set(s2.per_family_aucs) <= {"point_spike", "level_shift"}
    for fam_aucs in s2.per_family_aucs.values():
        for mid, a in fam_aucs.items():
            assert np.isfinite(a) and 0.0 <= a <= 1.0, f"bad AUC for {mid} = {a}"
    assert set(s2.per_model_score) == {mid for mid, _ in fitted}


@pytest.mark.smoke
def test_aggregation_borda_and_selection():
    s1 = {"a": 1.0, "b": 2.0, "c": 3.0}
    s2 = {"a": 2.0, "b": 1.0, "c": 3.0}
    s3 = {"a": 1.0, "b": 3.0, "c": 2.0}
    combined = borda_combine([s1, s2, s3])
    assert combined["a"] < combined["c"]  # 'a' is most consistently low rank
    out = select_from_ranks("series_x", "ms_pas", combined)
    assert out.selected == "a"
    # scores_to_ranks: higher_is_better
    ranks = scores_to_ranks({"a": 0.9, "b": 0.5, "c": 0.1}, higher_is_better=True)
    assert ranks["a"] < ranks["c"]


@pytest.mark.smoke
def test_kendall_agreement_well_defined():
    ranks_a = {"x": 1.0, "y": 2.0, "z": 3.0}
    ranks_b = {"x": 1.0, "y": 2.0, "z": 3.0}
    assert kendall_agreement(ranks_a, ranks_b) == pytest.approx(1.0)
    ranks_c = {"x": 3.0, "y": 2.0, "z": 1.0}
    assert kendall_agreement(ranks_a, ranks_c) == pytest.approx(-1.0)


@pytest.mark.smoke
def test_regret_computation_zero_when_perfect():
    oracle = {"a": 0.9, "b": 0.5, "c": 0.7}
    rec = compute_regret("s1", "perfect", "a", oracle)
    assert rec.regret == pytest.approx(0.0)
    rec_b = compute_regret("s1", "bad", "b", oracle)
    assert rec_b.regret == pytest.approx(0.4)
    summary = summarize_regret([rec, rec_b])
    assert summary["perfect"]["top1_hit_rate"] == 1.0
    assert summary["bad"]["top1_hit_rate"] == 0.0
