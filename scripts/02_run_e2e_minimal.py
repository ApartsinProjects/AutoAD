"""End-to-end pipeline run: Isolation Forest x 2 hyperparams on 2 datasets.

Single-model E2E to validate the full pipeline (load -> window -> fit ->
score -> per-point parquet -> metrics -> oracle aggregation) before
scaling up to the 60-candidate / 790-series production run.

Configuration
-------------
* Model: Isolation Forest, n_estimators in {50, 200}
* Datasets:
    - UCR Anomaly Archive 2021: first ``--ucr-limit`` series
      (default 20; full archive is 250)
    - Synthetic suite: ``--synth-per-family`` series per anomaly family
      across {point_spike, level_shift, trend_change, frequency_change}
      (default 5 per family = 20 total)
* Window length: 64, stride: 1
* Metrics: AUC-PR, AUC-ROC computed on per-point scores

Outputs
-------
* ``runs/oracle/per_point/{series_id}/{model_id}.parquet`` per (series, model)
* ``runs/oracle/oracle_e2e_v1.parquet`` aggregated metric table
* ``runs/manifests/e2e_<sha>.json`` reproducibility manifest

Usage::

    python scripts/02_run_e2e_minimal.py
    python scripts/02_run_e2e_minimal.py --ucr-limit 5 --synth-per-family 2  # tiny
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
from tqdm import tqdm

from autoad.data.loaders import iter_ucr
from autoad.data.synthetic_suite import build_suite
from autoad.data.windowing import scores_to_per_point, sliding_windows
from autoad.eval.metrics import auc_pr, auc_roc
from autoad.models.classical import IForest
from autoad.utils.io import (
    RUNS_DIR,
    config_hash,
    save_per_point_scores,
    write_manifest,
    write_parquet,
)

WINDOW = 64
STRIDE = 1
SYNTH_FAMILIES = ["point_spike", "level_shift", "trend_change", "frequency_change"]


def run_one(series, n_estimators: int, max_train_windows: int = 4000) -> dict:
    """Fit IForest on series.x_train (windowed), score series.x_test, return metrics."""
    series_id = series.series_id
    model_id = f"iforest_n_estimators={n_estimators}"

    if len(series.x_train) < WINDOW + 10 or len(series.x_test) < WINDOW + 10:
        return {
            "series_id": series_id,
            "source": series.source,
            "model_id": model_id,
            "status": "skipped_too_short",
        }

    t0 = time.perf_counter()
    train_windows = sliding_windows(series.x_train, WINDOW, STRIDE)
    test_windows = sliding_windows(series.x_test, WINDOW, STRIDE)
    # Cap training windows so OCSVM-grade fits stay tractable on long series.
    if len(train_windows) > max_train_windows:
        idx = np.linspace(0, len(train_windows) - 1, max_train_windows, dtype=int)
        train_windows = train_windows[idx]

    model = IForest(n_estimators=n_estimators, random_state=42).fit(train_windows)
    win_scores = model.score(test_windows)
    pp = scores_to_per_point(win_scores, len(series.x_test), WINDOW, STRIDE, reducer="max")
    elapsed = time.perf_counter() - t0

    # Persist per-point scores (with provenance metadata)
    save_per_point_scores(
        series_id,
        model_id,
        pp,
        extra_meta={
            "source": series.source,
            "n_estimators": str(n_estimators),
            "window": str(WINDOW),
            "stride": str(STRIDE),
            "fit_seconds": f"{elapsed:.3f}",
        },
    )

    return {
        "series_id": series_id,
        "source": series.source,
        "model_id": model_id,
        "n_estimators": n_estimators,
        "n_train_windows": int(len(train_windows)),
        "n_test_windows": int(len(test_windows)),
        "anom_fraction": float(series.y_test.mean()),
        "auc_pr": auc_pr(series.y_test, pp),
        "auc_roc": auc_roc(series.y_test, pp),
        "fit_seconds": elapsed,
        "status": "ok",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ucr-limit", type=int, default=20, help="number of UCR series")
    p.add_argument("--synth-per-family", type=int, default=5, help="synthetic series per family")
    p.add_argument(
        "--n-estimators",
        type=int,
        nargs="+",
        default=[50, 200],
        help="two IForest n_estimators values",
    )
    args = p.parse_args()

    cfg = {
        "model": "iforest",
        "n_estimators": args.n_estimators,
        "ucr_limit": args.ucr_limit,
        "synth_per_family": args.synth_per_family,
        "synth_families": SYNTH_FAMILIES,
        "window": WINDOW,
        "stride": STRIDE,
    }
    cfg_hash = config_hash(cfg)
    experiment_id = f"e2e_{cfg_hash}"
    print(f"Experiment ID: {experiment_id}")
    print(f"Config: {cfg}")
    print()

    # Build dataset list
    print("Loading UCR series...")
    ucr_series = list(iter_ucr(limit=args.ucr_limit))
    print(f"  Loaded {len(ucr_series)} UCR series")

    print("Building synthetic suite...")
    synth_series = build_suite(SYNTH_FAMILIES, n_per_family=args.synth_per_family)
    print(f"  Built {len(synth_series)} synthetic series ({len(SYNTH_FAMILIES)} families)")

    all_series = ucr_series + synth_series
    n_total_jobs = len(all_series) * len(args.n_estimators)
    print(f"\nRunning {len(all_series)} series x {len(args.n_estimators)} hyperparams = {n_total_jobs} fits\n")

    results: list[dict] = []
    t_start = time.perf_counter()
    pbar = tqdm(total=n_total_jobs, desc="E2E", unit="fit")
    for series in all_series:
        for n_est in args.n_estimators:
            row = run_one(series, n_est)
            results.append(row)
            pbar.update(1)
            pbar.set_postfix(series=series.series_id[:18], n_est=n_est)
    pbar.close()
    total_elapsed = time.perf_counter() - t_start

    # Aggregate into an oracle parquet
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] != "ok"]
    print(f"\nCompleted in {total_elapsed:.1f}s. OK={len(ok)} skipped={len(skipped)}")

    if not ok:
        print("ERROR: no successful runs", file=sys.stderr)
        return 1

    # Build oracle table
    cols = {
        "series_id": [r["series_id"] for r in ok],
        "source": [r["source"] for r in ok],
        "model_id": [r["model_id"] for r in ok],
        "n_estimators": [r["n_estimators"] for r in ok],
        "n_train_windows": [r["n_train_windows"] for r in ok],
        "n_test_windows": [r["n_test_windows"] for r in ok],
        "anom_fraction": [r["anom_fraction"] for r in ok],
        "auc_pr": [r["auc_pr"] for r in ok],
        "auc_roc": [r["auc_roc"] for r in ok],
        "fit_seconds": [r["fit_seconds"] for r in ok],
    }
    table = pa.table(cols)
    oracle_path = RUNS_DIR / "oracle" / f"oracle_{experiment_id}.parquet"
    write_parquet(oracle_path, table, meta={"experiment_id": experiment_id, "config_hash": cfg_hash})
    print(f"\nOracle table written: {oracle_path}")
    print(f"  rows: {len(ok)} ({len(cols)} columns)")

    manifest_path = write_manifest(
        experiment_id,
        cfg,
        outputs=[str(oracle_path.relative_to(RUNS_DIR.parent))],
    )
    print(f"Manifest written: {manifest_path}")

    # ------------------------------------------------------------------
    # Sanity report
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SANITY REPORT")
    print("=" * 78)

    aucs_finite = [r["auc_pr"] for r in ok if np.isfinite(r["auc_pr"])]
    rocs_finite = [r["auc_roc"] for r in ok if np.isfinite(r["auc_roc"])]
    print(f"\nFraction with finite AUC-PR: {len(aucs_finite)} / {len(ok)}")
    print(f"Fraction with finite AUC-ROC: {len(rocs_finite)} / {len(ok)}")

    if aucs_finite:
        print(f"\nAUC-PR  overall: mean={np.mean(aucs_finite):.4f}  median={np.median(aucs_finite):.4f}")
    if rocs_finite:
        print(f"AUC-ROC overall: mean={np.mean(rocs_finite):.4f}  median={np.median(rocs_finite):.4f}")

    # Per-(dataset, n_estimators) breakdown
    print("\nAUC-PR by (source, n_estimators):")
    print(f"{'source':<24} {'n_est':>6} {'n':>5}  {'mean':>7}  {'median':>7}  {'min':>7}  {'max':>7}")
    for src in ["ucr_anomaly_2021", "synthetic_v1"]:
        for n_est in args.n_estimators:
            cell = [r["auc_pr"] for r in ok if r["source"] == src and r["n_estimators"] == n_est and np.isfinite(r["auc_pr"])]
            if cell:
                print(
                    f"{src:<24} {n_est:>6} {len(cell):>5} "
                    f"{np.mean(cell):>7.3f}  {np.median(cell):>7.3f}  "
                    f"{np.min(cell):>7.3f}  {np.max(cell):>7.3f}"
                )
            else:
                print(f"{src:<24} {n_est:>6} {0:>5}  (no finite values)")

    # Per-family breakdown for synthetic
    print("\nAUC-PR by (synthetic family, n_estimators):")
    print(f"{'family':<20} {'n_est':>6} {'n':>5}  {'mean':>7}")
    for fam in SYNTH_FAMILIES:
        for n_est in args.n_estimators:
            cell = [
                r["auc_pr"] for r in ok
                if r["source"] == "synthetic_v1"
                and r["n_estimators"] == n_est
                and r["series_id"].startswith(f"synth_{fam}_")
                and np.isfinite(r["auc_pr"])
            ]
            if cell:
                print(f"{fam:<20} {n_est:>6} {len(cell):>5}  {np.mean(cell):>7.3f}")

    # Cross-hyperparameter agreement (sanity: n=50 and n=200 shouldn't disagree wildly)
    print("\nCross-hyperparameter Spearman (sanity, expect > 0.4):")
    n_lo, n_hi = args.n_estimators[0], args.n_estimators[1]
    paired_lo = {r["series_id"]: r["auc_pr"] for r in ok if r["n_estimators"] == n_lo and np.isfinite(r["auc_pr"])}
    paired_hi = {r["series_id"]: r["auc_pr"] for r in ok if r["n_estimators"] == n_hi and np.isfinite(r["auc_pr"])}
    common = sorted(set(paired_lo) & set(paired_hi))
    if len(common) >= 3:
        from scipy.stats import spearmanr
        rho, p = spearmanr([paired_lo[s] for s in common], [paired_hi[s] for s in common])
        print(f"  Spearman({n_lo} vs {n_hi}) = {rho:.3f} (p={p:.3g}, n={len(common)})")

    # Verify per-point parquet artifacts exist for a sample series
    sample = ok[0]
    sample_path = (
        RUNS_DIR / "oracle" / "per_point" / sample["series_id"] / f"{sample['model_id']}.parquet"
    )
    print(f"\nSample per-point artifact: {sample_path}")
    print(f"  exists: {sample_path.exists()}")
    if sample_path.exists():
        from autoad.utils.io import read_parquet_meta
        meta = read_parquet_meta(sample_path)
        print(f"  provenance keys: {sorted(meta)[:8]} ...")
        print(f"  code_sha_short: {meta.get('code_sha_short', '?')}")
        print(f"  timestamp_utc:  {meta.get('timestamp_utc', '?')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
