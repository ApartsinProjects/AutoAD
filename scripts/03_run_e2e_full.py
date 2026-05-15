"""Full E2E: oracle + Source 1 LCO (data + latent) + Source 2 + selection + regret.

Scope (kept small so the entire pipeline produces inspectable outputs):
* **Candidate pool**: 6 detectors = IForest(50, 200) + LOF(10, 20) + OCSVM(0.05, 0.1)
* **Datasets**:
    - UCR Anomaly Archive 2021: first ``--ucr-limit`` series (default 10)
    - Synthetic suite: 5 per family x 4 families = 20 series
* **Cluster configs** for LCO: KMeans + GMM at C in {4, 8} (4 configs)
* **Source 2 families**: point_spike / level_shift / trend_change / frequency_change

For every series this script produces:

1. Oracle row per candidate (AUC-PR, AUC-ROC against real labels)
2. Source 1 data-domain LCO rank per candidate
3. Source 1 latent-domain LCO rank per candidate
4. Source 2 synthetic-perturbation rank per candidate
5. Per-selector pick: random, default-iforest, ms-pas-borda
6. Selection regret vs oracle

Run with::

    python scripts/03_run_e2e_full.py --ucr-limit 10 --synth-per-family 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
from tqdm import tqdm

from autoad.data.features import PCAEncoder, summary_features
from autoad.data.loaders import iter_ucr
from autoad.data.synthetic import make_normal_sine
from autoad.data.synthetic_suite import build_suite
from autoad.data.windowing import aggregate_window_labels, scores_to_per_point, sliding_windows
from autoad.eval.metrics import auc_pr, auc_roc
from autoad.eval.regret import RegretRecord, compute_regret, summarize_regret
from autoad.models.classical import IForest, LOF, OCSVM
from autoad.pseudo.cluster_holdout import run_lco_one_series
from autoad.pseudo.synthetic_score import run_synthetic_source
from autoad.selection.aggregate import (
    borda_combine,
    kendall_agreement,
    scores_to_ranks,
    select_from_ranks,
)
from autoad.utils.io import (
    RUNS_DIR,
    config_hash,
    save_per_point_scores,
    write_manifest,
    write_parquet,
)

WINDOW = 64
STRIDE = 1
MAX_TRAIN_WINDOWS = 3000   # cap to keep classical fit times bounded
SYNTH_FAMILIES = ["point_spike", "level_shift", "trend_change", "frequency_change"]
CLUSTER_CONFIGS = [("kmeans", 4), ("kmeans", 8), ("gmm", 4), ("gmm", 8)]


def build_candidate_pool() -> list[tuple[str, callable]]:
    """Six-candidate pool as factories (so we can instantiate fresh copies per fold)."""
    return [
        ("iforest_n_estimators=50", lambda: IForest(n_estimators=50, random_state=42)),
        ("iforest_n_estimators=200", lambda: IForest(n_estimators=200, random_state=42)),
        ("lof_n_neighbors=10", lambda: LOF(n_neighbors=10)),
        ("lof_n_neighbors=20", lambda: LOF(n_neighbors=20)),
        ("ocsvm_nu=0.05", lambda: OCSVM(nu=0.05)),
        ("ocsvm_nu=0.10", lambda: OCSVM(nu=0.10)),
    ]


def run_oracle_for_series(series, candidate_pool):
    """Fit each candidate on series.x_train windows; score series.x_test windows.

    Returns:
        oracle_perfs : dict[model_id -> AUC-PR]
        fitted_models : list[(model_id, fitted_BaseAD)]
        details : list[dict] - one row per candidate with extra metrics
    """
    train_w = sliding_windows(series.x_train, WINDOW, STRIDE)
    if len(train_w) > MAX_TRAIN_WINDOWS:
        idx = np.linspace(0, len(train_w) - 1, MAX_TRAIN_WINDOWS, dtype=int)
        train_w = train_w[idx]
    test_w = sliding_windows(series.x_test, WINDOW, STRIDE)
    win_labels = aggregate_window_labels(series.y_test, WINDOW, STRIDE, mode="max")

    oracle_perfs: dict[str, float] = {}
    fitted_models: list[tuple[str, object]] = []
    details: list[dict] = []
    for model_id, factory in candidate_pool:
        t0 = time.perf_counter()
        try:
            model = factory().fit(train_w)
            win_scores = model.score(test_w)
            pp = scores_to_per_point(win_scores, len(series.x_test), WINDOW, STRIDE, reducer="max")
            apr = auc_pr(series.y_test, pp)
            aroc = auc_roc(series.y_test, pp)
            elapsed = time.perf_counter() - t0
            # Persist per-point scores
            save_per_point_scores(series.series_id, model_id, pp,
                                  extra_meta={"source": series.source, "fit_seconds": f"{elapsed:.3f}"})
            oracle_perfs[model_id] = float(apr) if np.isfinite(apr) else float("nan")
            fitted_models.append((model_id, model))
            details.append({
                "series_id": series.series_id,
                "source": series.source,
                "model_id": model_id,
                "auc_pr": oracle_perfs[model_id],
                "auc_roc": float(aroc) if np.isfinite(aroc) else float("nan"),
                "fit_seconds": elapsed,
            })
        except Exception as e:
            oracle_perfs[model_id] = float("nan")
            details.append({
                "series_id": series.series_id,
                "source": series.source,
                "model_id": model_id,
                "auc_pr": float("nan"),
                "auc_roc": float("nan"),
                "fit_seconds": float("nan"),
                "error": str(e)[:80],
            })
    return oracle_perfs, fitted_models, details, train_w


def run_lco_for_series(series, train_windows, candidate_pool, variant: str):
    """LCO with either data-domain summary features or PCA-latent features."""
    if variant == "data":
        features = summary_features(train_windows)
    elif variant == "latent":
        enc = PCAEncoder(n_components=16).fit(train_windows)
        features = enc.transform(train_windows)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Build unfitted instances for the candidate pool (used as templates)
    candidates_unfitted = [(mid, factory()) for mid, factory in candidate_pool]
    # E2E uses permissive bucket set so we see LCO output on all series
    # (production protocol filters trivial/impossible folds).
    return run_lco_one_series(
        series_id=series.series_id,
        train_windows=train_windows,
        features=features,
        candidate_pool=candidates_unfitted,
        cluster_configs=CLUSTER_CONFIGS,
        variant=variant,
        valid_buckets=("trivial", "easy", "medium", "hard"),
    )


def run_source2_for_series(series, fitted_models, normal_signal):
    return run_synthetic_source(
        series_id=series.series_id,
        fitted_models=fitted_models,
        normal_signal=normal_signal,
        window=WINDOW,
        stride=STRIDE,
        families=tuple(SYNTH_FAMILIES),
    )


def make_selectors_for_series(model_ids, lco_data, lco_latent, source2):
    """Construct rank dicts for each selector. Higher score -> lower rank."""
    selectors = {}

    # Source 1 data-domain: already a per-model-rank dict
    selectors["s1_data"] = lco_data.per_model_score  # lower = better (it's already rank mean)
    selectors["s1_latent"] = lco_latent.per_model_score  # lower = better

    # Source 2: per-model AUC (higher = better) -> convert to ranks
    s2_ranks = scores_to_ranks(source2.per_model_score, higher_is_better=True)
    selectors["s2"] = s2_ranks

    return selectors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ucr-limit", type=int, default=10)
    p.add_argument("--synth-per-family", type=int, default=5)
    args = p.parse_args()

    cfg = {
        "candidates": [m for m, _ in build_candidate_pool()],
        "ucr_limit": args.ucr_limit,
        "synth_per_family": args.synth_per_family,
        "synth_families": SYNTH_FAMILIES,
        "cluster_configs": CLUSTER_CONFIGS,
        "window": WINDOW,
        "stride": STRIDE,
        "max_train_windows": MAX_TRAIN_WINDOWS,
    }
    exp_id = f"e2e_full_{config_hash(cfg)}"
    print(f"Experiment ID: {exp_id}\nConfig: {cfg}\n")

    candidate_pool = build_candidate_pool()
    model_ids = [m for m, _ in candidate_pool]

    print("Loading UCR series...")
    ucr = list(iter_ucr(limit=args.ucr_limit))
    print("Building synthetic suite...")
    synth = build_suite(SYNTH_FAMILIES, n_per_family=args.synth_per_family)
    series_list = ucr + synth
    print(f"  Total: {len(series_list)} series")
    print()

    # Pre-generate a normal signal for Source 2 once per series source.
    # For UCR we use the second half of x_train (assumed clean).
    # For synthetic we just regenerate a clean sine of similar length.

    oracle_rows: list[dict] = []
    lco_records: list[dict] = []
    s2_records: list[dict] = []
    selector_rows: list[dict] = []
    regret_records: list[RegretRecord] = []
    inter_source_taus: list[dict] = []

    t0 = time.perf_counter()
    for series in tqdm(series_list, desc="series", unit="ser"):
        t_series = time.perf_counter()
        # 1) Oracle
        oracle_perfs, fitted, details, train_w = run_oracle_for_series(series, candidate_pool)
        oracle_rows.extend(details)
        if len(fitted) < 3:
            tqdm.write(f"  SKIP {series.series_id}: too few fitted models ({len(fitted)})")
            continue

        # 2) Source 1 LCO (data + latent)
        lco_data = run_lco_for_series(series, train_w, candidate_pool, "data")
        lco_latent = run_lco_for_series(series, train_w, candidate_pool, "latent")
        for variant, r in (("data", lco_data), ("latent", lco_latent)):
            for mid, score in r.per_model_score.items():
                lco_records.append({
                    "series_id": series.series_id, "variant": variant,
                    "model_id": mid, "lco_rank_mean": float(score),
                    "n_folds_used": r.n_folds_used,
                    "n_folds_excluded": r.n_folds_excluded,
                })

        # 3) Source 2 synthetic-perturbation scoring
        if series.source == "ucr_anomaly_2021":
            # Use the second half of x_train as a clean normal signal
            half = max(len(series.x_train) // 2, WINDOW * 4)
            normal_signal = series.x_train[half:].astype(np.float32)
        else:
            # Re-generate clean sine matched to synthetic build settings
            normal_signal = make_normal_sine(2000, period=50, noise=0.1, seed=999)
        s2 = run_source2_for_series(series, fitted, normal_signal)
        for fam, fam_aucs in s2.per_family_aucs.items():
            for mid, a in fam_aucs.items():
                s2_records.append({
                    "series_id": series.series_id, "family": fam,
                    "model_id": mid, "auc_pr": float(a),
                })

        # 4) Per-source ranks + combined Borda
        selectors_scores = make_selectors_for_series(model_ids, lco_data, lco_latent, s2)
        # Each entry is already a per-model rank (lower = better)
        per_source_ranks = {
            sel: dict(selectors_scores[sel]) for sel in ("s1_data", "s1_latent", "s2")
        }
        # Borda combine
        borda = borda_combine(list(per_source_ranks.values()))

        # 5) Build SelectorOutput records for each named selector
        rng = np.random.default_rng(42)
        # Random
        random_pick = str(rng.choice(model_ids))
        # Default = first IForest variant
        default_pick = "iforest_n_estimators=50"
        # MS-PAS combined (Borda)
        ms_pas = select_from_ranks(series.series_id, "ms_pas_borda", borda)
        # Source-1 data only
        s1_data_pick = select_from_ranks(series.series_id, "s1_data_only", per_source_ranks["s1_data"])
        # Source-1 latent only
        s1_latent_pick = select_from_ranks(series.series_id, "s1_latent_only", per_source_ranks["s1_latent"])
        # Source-2 only
        s2_pick = select_from_ranks(series.series_id, "s2_only", per_source_ranks["s2"])

        picks = [
            ("random", random_pick, float("nan")),
            ("default", default_pick, float("nan")),
            ("s1_data_only", s1_data_pick.selected, float("nan")),
            ("s1_latent_only", s1_latent_pick.selected, float("nan")),
            ("s2_only", s2_pick.selected, float("nan")),
            ("ms_pas_borda", ms_pas.selected, float("nan")),
        ]
        for sel_id, picked, conf in picks:
            selector_rows.append({
                "series_id": series.series_id, "selector_id": sel_id,
                "selected_model": picked, "confidence": conf,
            })
            regret_records.append(compute_regret(series.series_id, sel_id, picked, oracle_perfs))

        # 6) Inter-source agreement (Kendall) - sanity signal
        tau_dl = kendall_agreement(per_source_ranks["s1_data"], per_source_ranks["s1_latent"])
        tau_d2 = kendall_agreement(per_source_ranks["s1_data"], per_source_ranks["s2"])
        tau_l2 = kendall_agreement(per_source_ranks["s1_latent"], per_source_ranks["s2"])
        inter_source_taus.append({
            "series_id": series.series_id,
            "tau_data_latent": tau_dl,
            "tau_data_s2": tau_d2,
            "tau_latent_s2": tau_l2,
        })
        elapsed = time.perf_counter() - t_series
        tqdm.write(
            f"  {series.series_id}: oracle best={max(oracle_perfs, key=lambda m: oracle_perfs.get(m, -1))} "
            f"({oracle_perfs[max(oracle_perfs, key=lambda m: oracle_perfs.get(m, -1))]:.3f})  "
            f"MS-PAS pick={ms_pas.selected}  "
            f"folds (data/latent)={lco_data.n_folds_used}/{lco_latent.n_folds_used}  "
            f"{elapsed:.1f}s"
        )

    total = time.perf_counter() - t0
    print(f"\nFinished in {total:.1f}s.")

    # ------------------------------------------------------------------
    # Persist all artifacts
    # ------------------------------------------------------------------
    oracle_table = pa.table({k: [r.get(k, None) for r in oracle_rows] for k in
                              ["series_id", "source", "model_id", "auc_pr", "auc_roc", "fit_seconds"]})
    write_parquet(RUNS_DIR / "oracle" / f"oracle_{exp_id}.parquet", oracle_table,
                  meta={"experiment_id": exp_id})
    lco_table = pa.table({k: [r[k] for r in lco_records] for k in
                          ["series_id", "variant", "model_id", "lco_rank_mean",
                           "n_folds_used", "n_folds_excluded"]})
    write_parquet(RUNS_DIR / "lco" / f"lco_{exp_id}.parquet", lco_table,
                  meta={"experiment_id": exp_id})
    s2_table = pa.table({k: [r[k] for r in s2_records] for k in ["series_id", "family", "model_id", "auc_pr"]})
    write_parquet(RUNS_DIR / "source2" / f"s2_{exp_id}.parquet", s2_table,
                  meta={"experiment_id": exp_id})
    sel_table = pa.table({k: [r[k] for r in selector_rows] for k in
                          ["series_id", "selector_id", "selected_model", "confidence"]})
    write_parquet(RUNS_DIR / "selectors" / f"selectors_{exp_id}.parquet", sel_table,
                  meta={"experiment_id": exp_id})
    regret_table = pa.table({
        "series_id": [r.series_id for r in regret_records],
        "selector_id": [r.selector_id for r in regret_records],
        "selected_model": [r.selected_model for r in regret_records],
        "selected_perf": [r.selected_perf for r in regret_records],
        "oracle_best_model": [r.oracle_best_model for r in regret_records],
        "oracle_best_perf": [r.oracle_best_perf for r in regret_records],
        "regret": [r.regret for r in regret_records],
    })
    write_parquet(RUNS_DIR / "regret" / f"regret_{exp_id}.parquet", regret_table,
                  meta={"experiment_id": exp_id})
    write_manifest(exp_id, cfg, outputs=["oracle", "lco", "source2", "selectors", "regret"])

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("E2E REPORT")
    print("=" * 90)

    print(f"\nSeries processed: {len(set(r['series_id'] for r in oracle_rows))}")
    print(f"Oracle rows:     {len(oracle_rows)}")
    print(f"LCO rows:        {len(lco_records)}  (data + latent combined)")
    print(f"Source-2 rows:   {len(s2_records)}")
    print(f"Selector rows:   {len(selector_rows)}")
    print(f"Regret rows:     {len(regret_records)}")

    # Oracle: per-source mean AUC-PR
    print("\nOracle AUC-PR by source:")
    sources = sorted(set(r["source"] for r in oracle_rows))
    for src in sources:
        cell = [r["auc_pr"] for r in oracle_rows if r["source"] == src and np.isfinite(r["auc_pr"])]
        if cell:
            print(f"  {src:<24} n={len(cell):>3}  mean AUC-PR={np.mean(cell):.4f}  median={np.median(cell):.4f}")

    # Oracle: best-model frequency
    from collections import Counter
    oracle_best_per_series = {}
    for r in oracle_rows:
        sid = r["series_id"]
        cur = oracle_best_per_series.get(sid)
        if cur is None or (np.isfinite(r["auc_pr"]) and r["auc_pr"] > cur[1]):
            oracle_best_per_series[sid] = (r["model_id"], r["auc_pr"])
    best_counter = Counter(m for m, _ in oracle_best_per_series.values())
    print("\nOracle's best model frequency across series:")
    for mid, cnt in best_counter.most_common():
        print(f"  {mid:<32} {cnt}")

    # Inter-source Kendall tau distributions
    print("\nInter-source Kendall tau (mean across series; expect > 0 if sources agree):")
    keys = ["tau_data_latent", "tau_data_s2", "tau_latent_s2"]
    for k in keys:
        vals = [r[k] for r in inter_source_taus if np.isfinite(r[k])]
        if vals:
            print(f"  {k}: mean={np.mean(vals):+.3f}  median={np.median(vals):+.3f}  n={len(vals)}")
        else:
            print(f"  {k}: no finite values")

    # Selection-regret summary
    print("\nSelection regret (lower = better):")
    summary = summarize_regret(regret_records)
    print(f"{'selector':<18} {'n':>4}  {'mean_regret':>11}  {'median':>8}  {'top-1 hit':>9}")
    order = ["random", "default", "s1_data_only", "s1_latent_only", "s2_only", "ms_pas_borda"]
    for sel in order:
        if sel in summary:
            s = summary[sel]
            print(f"{sel:<18} {int(s['n']):>4}  {s['mean_regret']:>11.4f}  {s['median_regret']:>8.4f}  {s['top1_hit_rate']:>8.1%}")

    # Per-source ranks for first 2 series (so user can SEE rankings)
    print("\nPer-source ranks for sample series (lower = better detector):")
    sample_ids = sorted(set(r["series_id"] for r in oracle_rows))[:3]
    print(f"{'series':<22} {'model':<32} {'oracle_auc':>10} {'s1_data':>8} {'s1_lat':>8} {'s2':>8}")
    for sid in sample_ids:
        # Find rank info for this series across sources
        s1d = {r["model_id"]: r["lco_rank_mean"] for r in lco_records
               if r["series_id"] == sid and r["variant"] == "data"}
        s1l = {r["model_id"]: r["lco_rank_mean"] for r in lco_records
               if r["series_id"] == sid and r["variant"] == "latent"}
        s2map: dict[str, list[float]] = {}
        for r in s2_records:
            if r["series_id"] == sid:
                s2map.setdefault(r["model_id"], []).append(r["auc_pr"])
        s2avg = {m: float(np.mean([v for v in vs if np.isfinite(v)]) if any(np.isfinite(v) for v in vs) else np.nan)
                 for m, vs in s2map.items()}
        oracle_aucs = {r["model_id"]: r["auc_pr"] for r in oracle_rows if r["series_id"] == sid}
        for mid in model_ids:
            print(f"{sid:<22} {mid:<32} "
                  f"{oracle_aucs.get(mid, float('nan')):>10.3f} "
                  f"{s1d.get(mid, float('nan')):>8.2f} "
                  f"{s1l.get(mid, float('nan')):>8.2f} "
                  f"{s2avg.get(mid, float('nan')):>8.3f}")
        print()

    print("=" * 90)
    print("Artifacts written under runs/:  oracle/, lco/, source2/, selectors/, regret/")
    print(f"Manifest: runs/manifests/{exp_id}.json")
    print(f"Wall clock: {total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
