# AutoAD

Multi-Source Pseudo-Anomaly Selection for unsupervised time-series anomaly detection model selection.

* **Paper blueprint**: <https://apartsinprojects.github.io/AutoAD/>
* **Research plan**: [`plan/anomaly_detection_model_selection_plan.md`](plan/anomaly_detection_model_selection_plan.md)
* **Implementation plan**: [`implementation plan.md`](implementation%20plan.md)

## Quick start

```bash
# Python 3.11 is required (3.12+ also works)
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\activate on cmd.exe
pip install --upgrade pip wheel

# Install CUDA 12.1 PyTorch first (skip the index-url to install CPU-only torch)
pip install --index-url https://download.pytorch.org/whl/cu121 "torch>=2.2,<2.5"

# Install the project and remaining deps
pip install -e .

# Download UCR Anomaly Archive 2021 (~184 MB)
python scripts/01_download_ucr.py

# Run smoke tests (under 30 s on a laptop)
pytest tests/smoke/ -v
```

## Repository layout

```
src/autoad/
  data/            # loaders, synthetic generators, windowing
  models/          # BaseAD interface + classical detectors (more in Phase 2)
  pseudo/          # Phase 3: Leave-Cluster-Out + synthetic injections
  selection/       # Phase 4: type-aware rank aggregation
  eval/            # VUS-PR, AUC-PR, range-F1, regret
  robustness/      # Phase 7: confidence ROC, within-eps, family confusion
  utils/           # IO, caching, provenance
scripts/           # 01..09 phase entry points
tests/smoke/       # phase smoke tests (one per phase)
configs/           # Hydra configs (datasets, models, experiments)
data/              # gitignored; populated by download scripts
runs/              # gitignored; experimental artifacts (parquet, mlflow)
vendored/          # gitignored; cloned TSB-AD, Goswami, MSAD upstreams
```

## Status

Phase 0 (infrastructure) and Phase 1 (data ingestion) are complete:

* ✓ `BaseAD` interface with ECDF score normalization
* ✓ Three reference detectors (IForest, LOF, OCSVM) with hyperparameter grids
* ✓ Synthetic anomaly generators for five families (point, level shift, trend, frequency, contextual)
* ✓ Sliding-window + per-point reduction utilities
* ✓ Per-point score parquet persistence with provenance
* ✓ UCR Anomaly Archive 2021 loader (250 series)
* ✓ 22 smoke tests passing in 15 s
* ✓ Vendored TSB-AD, Goswami (tsad-model-selection), MSAD

Next: Phase 2 (full 60-candidate oracle on real benchmarks), Phase 3 (LCO + synthetic + residual pseudo-anomaly sources).

See [`implementation plan.md`](implementation%20plan.md) Section 9 for the week-by-week schedule.
