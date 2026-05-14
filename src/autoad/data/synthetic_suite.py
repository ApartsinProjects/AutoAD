"""Materialized synthetic series with proper train/test split.

For AD model selection we need (x_train, x_test, y_test). The
generators in :mod:`autoad.data.synthetic` return a perturbed full signal
plus per-point labels. This module adds the train/test split convention
used throughout the pipeline:

* First half of a length-2000 series = clean normal training data
* Second half = test signal with injected anomalies + binary labels

Each synthetic series has a deterministic ``series_id`` of the form
``"synth_<family>_<idx>"`` so artifacts are reproducible across runs.
"""

from __future__ import annotations

import numpy as np

from .loaders import Series
from .synthetic import INJECTORS, inject_contextual, make_normal_sine


def make_synthetic_series(
    family: str,
    idx: int,
    length: int = 2000,
    period: int = 50,
    noise: float = 0.1,
    seed_offset: int = 0,
) -> Series:
    """Build one synthetic series with a clean train half and anomalous test half."""
    seed = idx * 1000 + seed_offset
    x = make_normal_sine(length, period=period, noise=noise, seed=seed)
    train_len = length // 2
    x_train = x[:train_len].copy()
    x_test_clean = x[train_len:].copy()

    if family == "contextual":
        x_test, y_test = inject_contextual(x_test_clean, period=period, seed=seed + 1)
    elif family in INJECTORS:
        x_test, y_test = INJECTORS[family](x_test_clean, seed=seed + 1)
    else:
        raise ValueError(f"Unknown family: {family!r}")

    return Series(
        series_id=f"synth_{family}_{idx:03d}",
        x_train=x_train,
        x_test=x_test,
        y_test=y_test,
        source="synthetic_v1",
        meta={
            "family": family,
            "idx": idx,
            "length": length,
            "period": period,
            "noise": noise,
            "seed": seed,
            "anom_fraction": float(y_test.mean()),
        },
    )


def build_suite(
    families: list[str],
    n_per_family: int = 5,
    length: int = 2000,
    period: int = 50,
    noise: float = 0.1,
) -> list[Series]:
    """Build a list of synthetic series spanning the requested families."""
    out: list[Series] = []
    for fam in families:
        for i in range(n_per_family):
            out.append(make_synthetic_series(fam, i, length=length, period=period, noise=noise))
    return out
