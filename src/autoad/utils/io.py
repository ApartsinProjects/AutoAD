"""I/O helpers for AutoAD artifacts.

All experimental artifacts live under ``runs/`` and are written as
parquet (tabular) or numpy (raw arrays) with a fixed schema version in
their filename. Every artifact carries a provenance header (code SHA,
config hash, seeds, library versions, timestamp).

Schemas
-------
* ``oracle`` parquet: one row per ``(series_id, model_id, metric)``;
  columns ``series_id: str, model_id: str, metric: str, value: float32``.
* ``per_point`` parquet: one row per time step;
  columns ``t: int32, score: float32``.
* ``ranks`` parquet: one row per model in a per-source ranking;
  columns ``model_id: str, rank: int32, source: str``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Repo root: this file lives at src/autoad/utils/io.py
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "runs"


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------

def git_sha(short: bool = False) -> str:
    """Return current HEAD SHA, or ``'nogit'`` if not in a repo."""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        out = subprocess.run(
            cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=5, check=True
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "nogit"


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable hash of a config dict; uses canonical JSON."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Standard provenance metadata attached to every artifact."""
    meta: dict[str, Any] = {
        "code_sha": git_sha(),
        "code_sha_short": git_sha(short=True),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        meta.update(extra)
    return meta


# ----------------------------------------------------------------------
# Parquet I/O with atomic write + provenance header
# ----------------------------------------------------------------------

def write_parquet(
    path: Path,
    table: pa.Table | dict[str, Any],
    meta: dict[str, Any] | None = None,
    compression: str = "zstd",
) -> Path:
    """Atomically write a parquet file with provenance in schema metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(table, dict):
        table = pa.table(table)
    full_meta = provenance(meta or {})
    schema_meta = {k.encode(): str(v).encode() for k, v in full_meta.items()}
    existing = dict(table.schema.metadata or {})
    existing.update(schema_meta)
    table = table.replace_schema_metadata(existing)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression=compression)
    os.replace(tmp, path)
    return path


def read_parquet(path: Path) -> pa.Table:
    return pq.read_table(Path(path))


def read_parquet_meta(path: Path) -> dict[str, str]:
    """Read just the schema-level provenance metadata."""
    schema = pq.read_schema(Path(path))
    raw = schema.metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items()}


# ----------------------------------------------------------------------
# Manifest: per-experiment lock file
# ----------------------------------------------------------------------

def write_manifest(experiment_id: str, cfg: dict[str, Any], outputs: list[str]) -> Path:
    """Write a manifest under runs/manifests/{experiment_id}.json."""
    manifest_dir = RUNS_DIR / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{experiment_id}.json"
    payload = {
        "experiment_id": experiment_id,
        "config_hash": config_hash(cfg),
        "config": cfg,
        "outputs": outputs,
        **provenance(),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


# ----------------------------------------------------------------------
# Per-point score storage
# ----------------------------------------------------------------------

def save_per_point_scores(
    series_id: str,
    model_id: str,
    scores: np.ndarray,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Persist a 1-D score time series to ``runs/oracle/per_point/{series}/{model}.parquet``.

    Storing the per-point score series (not just summary metrics) means
    any future metric (VUS-PR, AUC-PR, range-F1) can be recomputed without
    retraining the model.
    """
    path = RUNS_DIR / "oracle" / "per_point" / series_id / f"{model_id}.parquet"
    t = np.arange(len(scores), dtype=np.int32)
    table = pa.table({"t": t, "score": np.asarray(scores, dtype=np.float32)})
    return write_parquet(path, table, meta={"series_id": series_id, "model_id": model_id, **(extra_meta or {})})


def load_per_point_scores(series_id: str, model_id: str) -> np.ndarray:
    path = RUNS_DIR / "oracle" / "per_point" / series_id / f"{model_id}.parquet"
    table = read_parquet(path)
    return table.column("score").to_numpy()


def per_point_exists(series_id: str, model_id: str) -> bool:
    return (RUNS_DIR / "oracle" / "per_point" / series_id / f"{model_id}.parquet").exists()
