"""Feature extractors for clustering windows.

Two extractors are provided so we can study both LCO variants:

* :func:`summary_features` — fast numpy-vectorized panel of statistical /
  spectral / temporal descriptors (data-domain LCO).
* :class:`PCAEncoder` — PCA-bottleneck of raw windows (latent-domain LCO).
  PCA acts as a deliberately simple, content-agnostic encoder; in later
  phases it is replaced by an autoencoder or TS2Vec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


# Names align with the feature index in :func:`summary_features` output.
SUMMARY_FEATURE_NAMES = [
    "mean", "std", "min", "max",
    "skew", "kurt",
    "ac_lag1", "ac_lag2", "ac_lag5", "ac_lag10",
    "trend_slope",
    "spec_entropy", "dom_freq",
    "ptp", "median_abs_dev",
]


def _autocorr(x: np.ndarray, lag: int) -> np.ndarray:
    """Lag-k autocorrelation per row of a 2D ``(n, w)`` array."""
    if lag >= x.shape[1]:
        return np.zeros(x.shape[0], dtype=np.float64)
    a = x[:, :-lag]
    b = x[:, lag:]
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    num = (a_c * b_c).sum(axis=1)
    den = np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum(axis=1)) + 1e-12
    return num / den


def _trend_slope(x: np.ndarray) -> np.ndarray:
    """Slope of a least-squares linear fit per row."""
    w = x.shape[1]
    t = np.arange(w, dtype=np.float64)
    t_mean = t.mean()
    t_centered = t - t_mean
    x_centered = x - x.mean(axis=1, keepdims=True)
    return (x_centered @ t_centered) / ((t_centered ** 2).sum() + 1e-12)


def _spectral_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (spectral_entropy, dominant_frequency_index) per row."""
    fft = np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1)
    power = np.abs(fft) ** 2
    psum = power.sum(axis=1, keepdims=True) + 1e-12
    p = power / psum
    # Spectral entropy (Shannon) of normalized power spectrum.
    p_safe = np.clip(p, 1e-12, 1.0)
    entropy = -(p * np.log(p_safe)).sum(axis=1)
    dom = p[:, 1:].argmax(axis=1).astype(np.float64)  # skip DC
    return entropy, dom


def summary_features(windows: np.ndarray) -> np.ndarray:
    """Compute a 15-feature panel for each row of a ``(n, w)`` array.

    Returns ``(n, 15)`` float32 matrix; feature names are in
    :data:`SUMMARY_FEATURE_NAMES`. Vectorized over the window axis;
    no Python-level loops over windows.
    """
    if windows.ndim != 2:
        raise ValueError(f"Expected 2D windows; got shape {windows.shape}")
    x = np.asarray(windows, dtype=np.float64)
    n = x.shape[0]
    out = np.zeros((n, 15), dtype=np.float64)
    out[:, 0] = x.mean(axis=1)
    out[:, 1] = x.std(axis=1)
    out[:, 2] = x.min(axis=1)
    out[:, 3] = x.max(axis=1)
    # Skew/kurt: use centered moments
    mu = out[:, 0:1]
    sigma = out[:, 1:2] + 1e-12
    z = (x - mu) / sigma
    out[:, 4] = (z ** 3).mean(axis=1)
    out[:, 5] = (z ** 4).mean(axis=1) - 3.0
    out[:, 6] = _autocorr(x, 1)
    out[:, 7] = _autocorr(x, 2)
    out[:, 8] = _autocorr(x, 5)
    out[:, 9] = _autocorr(x, 10)
    out[:, 10] = _trend_slope(x)
    sp_ent, dom = _spectral_features(x)
    out[:, 11] = sp_ent
    out[:, 12] = dom
    out[:, 13] = x.max(axis=1) - x.min(axis=1)
    out[:, 14] = np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1)
    # Replace any non-finite cells (e.g. flat windows) with 0
    out = np.where(np.isfinite(out), out, 0.0)
    return out.astype(np.float32)


# --------------------------------------------------------------------
# Latent encoder: PCA bottleneck
# --------------------------------------------------------------------

@dataclass
class PCAEncoder:
    """Train PCA on a stack of windows; project new windows to latent space.

    The encoder is fit ONCE on the full ``X_N`` (normal training windows)
    of a series and shared across all LCO folds, to avoid the circularity
    in which the encoder is retrained per fold and could memorize the
    holdout structure.
    """

    n_components: int = 16
    random_state: int = 42

    def __post_init__(self) -> None:
        self._pca: PCA | None = None

    def fit(self, windows: np.ndarray) -> "PCAEncoder":
        if windows.ndim != 2:
            raise ValueError(f"Expected 2D windows; got shape {windows.shape}")
        # Cap n_components by both n_windows and feature dim.
        nc = min(self.n_components, windows.shape[0] - 1, windows.shape[1])
        self._pca = PCA(n_components=max(nc, 2), random_state=self.random_state)
        self._pca.fit(windows.astype(np.float64))
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCAEncoder: must call fit() before transform()")
        return self._pca.transform(windows.astype(np.float64)).astype(np.float32)

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        return self.fit(windows).transform(windows)
