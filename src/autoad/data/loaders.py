"""Dataset loaders for AutoAD benchmarks.

Each loader returns a :class:`Series` dataclass:

* ``series_id``: unique string ID (used for artifact paths)
* ``x_train``: 1D normal-only training signal (float32)
* ``x_test``: 1D test signal containing both normals and anomalies
* ``y_test``: 1D binary label aligned with ``x_test``
* ``source``: benchmark identifier ("ucr_anomaly_2021", etc.)
* ``meta``: dict of metadata (anomaly range, dataset name, etc.)

The UCR Anomaly Archive 2021 encodes train/test split in the filename:
``<num>_UCR_Anomaly_<name>_<train_end>_<anom_start>_<anom_end>.txt``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
UCR_DIR = REPO_ROOT / "data" / "raw" / "UCR_Anomaly_2021"

# Regex captures train_end, anom_start, anom_end from the filename.
# Example: "001_UCR_Anomaly_DISTORTEDECG_4000_5800_6000.txt"
UCR_NAME_RE = re.compile(
    r"^(?P<num>\d+)_UCR_Anomaly_(?P<name>.+?)_"
    r"(?P<train_end>\d+)_(?P<anom_start>\d+)_(?P<anom_end>\d+)\.txt$"
)


@dataclass
class Series:
    """A single labeled anomaly-detection series."""

    series_id: str
    x_train: np.ndarray  # 1-D float32, normal only
    x_test: np.ndarray   # 1-D float32, mixed normal + anomalous
    y_test: np.ndarray   # 1-D uint8, 0/1 aligned with x_test
    source: str
    meta: dict[str, Any] = field(default_factory=dict)


def _read_ucr_file(path: Path) -> np.ndarray:
    """UCR files are either one-value-per-line or space-separated. Be lenient."""
    txt = path.read_text(errors="replace").strip()
    # Try whitespace split (handles single-line space-sep and multi-line)
    arr = np.fromstring(txt.replace("\n", " "), sep=" ", dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{path}: parsed as empty")
    return arr.astype(np.float32)


def load_ucr_series(path: Path) -> Series:
    """Parse one UCR Anomaly Archive 2021 file."""
    m = UCR_NAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected UCR filename: {path.name}")
    train_end = int(m["train_end"])
    anom_start = int(m["anom_start"])
    anom_end = int(m["anom_end"])
    name = m["name"]
    num = int(m["num"])
    x = _read_ucr_file(path)
    n = len(x)
    if not (0 < train_end < anom_start <= anom_end <= n):
        raise ValueError(
            f"{path.name}: inconsistent indices "
            f"(train_end={train_end}, anom=[{anom_start},{anom_end}], len={n})"
        )
    x_train = x[:train_end].copy()
    x_test = x[train_end:].copy()
    y_test = np.zeros(len(x_test), dtype=np.uint8)
    # Anomaly indices in the original series; shift to test-relative
    s_rel = max(anom_start - train_end, 0)
    e_rel = min(anom_end - train_end, len(x_test))
    y_test[s_rel:e_rel] = 1
    return Series(
        series_id=f"ucr_{num:03d}",
        x_train=x_train,
        x_test=x_test,
        y_test=y_test,
        source="ucr_anomaly_2021",
        meta={
            "num": num,
            "name": name,
            "train_end": train_end,
            "anom_start": anom_start,
            "anom_end": anom_end,
            "filename": path.name,
        },
    )


def iter_ucr(limit: int | None = None) -> Iterator[Series]:
    """Yield UCR series; ``limit`` truncates for smoke testing."""
    if not UCR_DIR.exists():
        raise FileNotFoundError(
            f"UCR data not found at {UCR_DIR}. Run scripts/01_download_ucr.py first."
        )
    paths = sorted(UCR_DIR.glob("*.txt"))
    if limit is not None:
        paths = paths[:limit]
    for p in paths:
        try:
            yield load_ucr_series(p)
        except Exception as e:
            # Don't fail the whole iteration on one bad file; log and skip.
            print(f"WARN: failed to load {p.name}: {e}")
            continue


def list_ucr_files() -> list[Path]:
    if not UCR_DIR.exists():
        return []
    return sorted(UCR_DIR.glob("*.txt"))
