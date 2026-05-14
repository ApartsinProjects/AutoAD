"""Sliding-window utilities for time-series AD.

AD detectors typically operate on fixed-length windows. This module
provides numpy-vectorized window extraction and per-window label
aggregation.
"""

from __future__ import annotations

import numpy as np


def sliding_windows(
    x: np.ndarray,
    window: int,
    stride: int = 1,
) -> np.ndarray:
    """Return windows of shape ``(n_windows, window)``.

    Uses ``numpy.lib.stride_tricks.sliding_window_view`` for zero-copy
    construction. Strided result is materialized as a contiguous array
    only if the caller passes the output downstream to a function that
    requires it.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"sliding_windows expects 1D input; got shape {x.shape}")
    if window > len(x):
        raise ValueError(f"window={window} > len(x)={len(x)}")
    sw = np.lib.stride_tricks.sliding_window_view(x, window)
    return sw[::stride].copy()  # copy avoids strided-write surprises downstream


def aggregate_window_labels(
    labels: np.ndarray,
    window: int,
    stride: int = 1,
    mode: str = "max",
) -> np.ndarray:
    """Aggregate per-point binary labels into per-window labels.

    ``mode="max"``: 1 if any point in the window is anomalous (standard).
    ``mode="mean"``: average label (used for soft labels).
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be 1D")
    sw = np.lib.stride_tricks.sliding_window_view(labels, window)[::stride]
    if mode == "max":
        return sw.max(axis=1).astype(np.uint8)
    if mode == "mean":
        return sw.mean(axis=1).astype(np.float32)
    raise ValueError(f"unknown mode {mode!r}")


def scores_to_per_point(
    window_scores: np.ndarray,
    series_len: int,
    window: int,
    stride: int = 1,
    reducer: str = "max",
) -> np.ndarray:
    """Map per-window scores back to per-point scores of length ``series_len``.

    Each point may belong to multiple windows (when stride < window).
    ``reducer="max"`` takes the maximum score among containing windows;
    ``reducer="mean"`` averages.

    Returns a float32 array of length ``series_len``.
    """
    n_windows = len(window_scores)
    expected = (series_len - window) // stride + 1
    if n_windows != expected:
        raise ValueError(
            f"got {n_windows} window scores; expected {expected} "
            f"for series_len={series_len}, window={window}, stride={stride}"
        )
    out = np.full(series_len, np.nan, dtype=np.float32)
    counts = np.zeros(series_len, dtype=np.int32)
    if reducer == "max":
        out[:] = -np.inf
        for i, s in enumerate(window_scores):
            start = i * stride
            out[start : start + window] = np.maximum(out[start : start + window], s)
        # Any uncovered points -> 0 (shouldn't happen if stride <= window)
        out = np.where(np.isfinite(out), out, 0.0)
        return out.astype(np.float32)
    if reducer == "mean":
        accum = np.zeros(series_len, dtype=np.float64)
        for i, s in enumerate(window_scores):
            start = i * stride
            accum[start : start + window] += s
            counts[start : start + window] += 1
        counts = np.where(counts == 0, 1, counts)
        return (accum / counts).astype(np.float32)
    raise ValueError(f"unknown reducer {reducer!r}")
