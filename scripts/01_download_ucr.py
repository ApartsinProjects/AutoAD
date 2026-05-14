"""Download the UCR Anomaly Archive 2021 (250 univariate single-anomaly series).

Source: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

The archive ships as a single ~260 MB zip containing files named:
   ``<num>_UCR_Anomaly_<dataset>_<train_end>_<anom_start>_<anom_end>.txt``

Each file contains a 1-D series; the train/test split and the
anomaly range are encoded in the filename. The loader (``autoad.data.loaders``)
parses that filename to recover the labels.

Usage::

    python scripts/01_download_ucr.py
    python scripts/01_download_ucr.py --force  # re-download even if extracted
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "UCR_Anomaly_2021"
ZIP_PATH = REPO_ROOT / "data" / "raw" / "UCR_Anomaly_2021.zip"

# Primary and Figshare mirror URLs. Try primary first, fall back to Figshare.
URLS = [
    "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip",
    "https://figshare.com/ndownloader/files/48125345",
]


def download(url: str, dest: Path, chunk: int = 1 << 20) -> bool:
    """Stream-download with progress; return True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [{url}] FAILED: {e}", file=sys.stderr)
        return False
    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for buf in resp.iter_content(chunk_size=chunk):
            fh.write(buf)
            pbar.update(len(buf))
    os.replace(tmp, dest)
    return True


def extract(zip_path: Path, dest_dir: Path) -> int:
    """Unzip, flattening the archive into ``dest_dir``. Return file count."""
    import zipfile
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".txt") and "UCR_Anomaly" in m]
        for m in tqdm(members, desc="extracting"):
            target = dest_dir / Path(m).name
            with zf.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
            count += 1
    return count


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="re-download even if present")
    args = p.parse_args()

    if DATA_DIR.exists() and not args.force:
        files = list(DATA_DIR.glob("*.txt"))
        if len(files) >= 240:
            print(f"Already extracted: {len(files)} files in {DATA_DIR}")
            return 0

    if not ZIP_PATH.exists() or args.force:
        downloaded = False
        for url in URLS:
            print(f"Trying {url}")
            if download(url, ZIP_PATH):
                downloaded = True
                break
        if not downloaded:
            print("All download URLs failed.", file=sys.stderr)
            return 1
    else:
        print(f"Zip already cached: {ZIP_PATH}")

    print(f"SHA-256: {sha256(ZIP_PATH)}")
    n = extract(ZIP_PATH, DATA_DIR)
    print(f"Extracted {n} files to {DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
