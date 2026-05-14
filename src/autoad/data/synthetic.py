"""Controlled synthetic anomaly generators.

Implements the six anomaly families specified in Section 3.2 of the
implementation plan. Each generator takes a normal signal and returns
(perturbed_signal, anomaly_labels) where ``anomaly_labels`` is a binary
mask aligned with the input length.

Families
--------
* ``point_spike``: multiply isolated point(s) by k*sigma
* ``level_shift``: add c*sigma offset over a contiguous window
* ``trend_change``: add linear slope over a window
* ``frequency_change``: replace window with same-mean signal at altered frequency
* ``contextual``: swap a window into wrong seasonal phase
* ``mode_exclusion``: train on K-1 modes, test on held-out mode (handled at dataset level)
"""

from __future__ import annotations

import numpy as np


def make_normal_sine(
    length: int,
    period: int = 50,
    noise: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Generate a simple sinusoidal "normal" signal with Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    base = np.sin(2 * np.pi * t / period)
    return base + rng.normal(0, noise, size=length).astype(np.float32)


def inject_point_spike(
    x: np.ndarray,
    n_anomalies: int = 3,
    k_sigma: float = 5.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.asarray(x, dtype=np.float32).copy()
    labels = np.zeros(len(y), dtype=np.uint8)
    sigma = float(y.std())
    # Avoid the first/last 1% of points so windowed scoring covers them
    margin = max(int(0.01 * len(y)), 1)
    idxs = rng.choice(np.arange(margin, len(y) - margin), size=n_anomalies, replace=False)
    for i in idxs:
        sign = rng.choice([-1, 1])
        y[i] = y[i] + sign * k_sigma * sigma
        labels[i] = 1
    return y, labels


def inject_level_shift(
    x: np.ndarray,
    n_anomalies: int = 2,
    width_frac: float = 0.05,
    c_sigma: float = 4.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.asarray(x, dtype=np.float32).copy()
    labels = np.zeros(len(y), dtype=np.uint8)
    sigma = float(y.std())
    w = max(int(width_frac * len(y)), 5)
    for _ in range(n_anomalies):
        start = int(rng.integers(0, len(y) - w))
        sign = rng.choice([-1, 1])
        y[start : start + w] += sign * c_sigma * sigma
        labels[start : start + w] = 1
    return y, labels


def inject_trend_change(
    x: np.ndarray,
    n_anomalies: int = 1,
    width_frac: float = 0.10,
    slope_sigma: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.asarray(x, dtype=np.float32).copy()
    labels = np.zeros(len(y), dtype=np.uint8)
    sigma = float(y.std())
    w = max(int(width_frac * len(y)), 10)
    for _ in range(n_anomalies):
        start = int(rng.integers(0, len(y) - w))
        slope = slope_sigma * sigma
        ramp = np.arange(w, dtype=np.float32) * slope
        y[start : start + w] += ramp
        labels[start : start + w] = 1
    return y, labels


def inject_frequency_change(
    x: np.ndarray,
    n_anomalies: int = 1,
    width_frac: float = 0.10,
    freq_mult: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.asarray(x, dtype=np.float32).copy()
    labels = np.zeros(len(y), dtype=np.uint8)
    w = max(int(width_frac * len(y)), 10)
    for _ in range(n_anomalies):
        start = int(rng.integers(0, len(y) - w))
        # Replace with a faster sinusoid scaled to the local std and mean.
        local_mean = float(y[start : start + w].mean())
        local_std = float(y[start : start + w].std()) + 1e-6
        t = np.arange(w, dtype=np.float32)
        replacement = local_mean + local_std * np.sin(2 * np.pi * freq_mult * t / w)
        y[start : start + w] = replacement.astype(np.float32)
        labels[start : start + w] = 1
    return y, labels


def inject_contextual(
    x: np.ndarray,
    period: int,
    n_anomalies: int = 2,
    width_frac: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Swap a window into the wrong seasonal phase (period/2 offset)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(x, dtype=np.float32).copy()
    labels = np.zeros(len(y), dtype=np.uint8)
    w = max(int(width_frac * len(y)), 5)
    half = period // 2
    for _ in range(n_anomalies):
        start = int(rng.integers(half, len(y) - w - half))
        # Copy values from period/2 earlier (wrong phase, same magnitude regime)
        y[start : start + w] = y[start - half : start - half + w]
        labels[start : start + w] = 1
    return y, labels


INJECTORS = {
    "point_spike": inject_point_spike,
    "level_shift": inject_level_shift,
    "trend_change": inject_trend_change,
    "frequency_change": inject_frequency_change,
    "contextual": inject_contextual,
}


def make_synthetic_series(
    family: str,
    length: int = 1000,
    period: int = 50,
    noise: float = 0.1,
    seed: int = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """One-shot helper: produce ``(perturbed_signal, labels)`` for a family."""
    x = make_normal_sine(length, period=period, noise=noise, seed=seed)
    if family == "contextual":
        return inject_contextual(x, period=period, seed=seed, **kwargs)
    if family not in INJECTORS:
        raise ValueError(f"Unknown family {family!r}; valid: {sorted(INJECTORS)}")
    return INJECTORS[family](x, seed=seed, **kwargs)
