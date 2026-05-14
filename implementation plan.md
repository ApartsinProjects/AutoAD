# AutoAD Implementation Plan
## Multi-Source Pseudo-Anomaly Validation for Unsupervised Time-Series AD Model Selection

**Version**: 1.0 (2026-05-14)
**Companion to**: [plan/anomaly_detection_model_selection_plan.md](plan/anomaly_detection_model_selection_plan.md)
**Target venue (primary)**: NeurIPS 2026 (abstract deadline mid-May 2026 already past, full deadline typically Aug). Realistic targets: KDD 2027 (Feb 2027), ICML 2027 (Jan 2027), ICDM 2026 (Jun 2026).

---

## 0. TL;DR

We build **MS-PAS** (Multi-Source Pseudo-Anomaly Selection), the first normal-data-only model-selection protocol for time-series anomaly detection that (i) introduces **Leave-Cluster-Out (LCO)** validation as a new pseudo-anomaly source, (ii) aggregates heterogeneous pseudo-anomaly signals (LCO at multiple granularities, six synthetic perturbation families, prediction-residual consistency) via anomaly-type-aware rank aggregation, (iii) provides a **selection-regret bound** under a stated manifold-margin condition, and (iv) demonstrates a **≥3% absolute VUS-PR regret reduction over every zero-label baseline** (Goswami ICLR 2023, N-1 Experts AutoML 2022, Idan ECAI 2024, IREOS-fast TKDD 2024, ASOI 2025) across TSB-UAD + UCR Anomaly Archive + controlled synthetic suites, with MSAD/Choose Wisely (PVLDB 2023) serving as the meta-supervised upper reference. The deliverable is one breakthrough-grade paper.

---

## 1. Sharpened Thesis and Differentiation

### 1.1 Refined Central Claim

> The internal cluster geometry of normal data, combined with structured synthetic perturbations of that data, contains enough information to rank anomaly detectors to within a small AUC gap of an oracle selector that uses real anomaly labels, **provided the pseudo-anomaly mixture spans the manifold-margin regimes that real anomalies populate**. The required mixture, the manifold-margin condition, and the failure modes are characterizable from the normal data alone.

### 1.2 What is genuinely new

| Component | Novel? | What prior work does | What we do |
|---|---|---|---|
| Leave-Cluster-Out (LCO) validation for AD | **Yes (no published precedent)** | Cluster-based OOD eval exists in vision; AD selection uses synthetic injections only | Held-out normal subdomains as mode-exclusion pseudo-anomalies, multi-granularity, difficulty-stratified |
| Multi-source pseudo-anomaly aggregation | **Yes (extends Goswami 2023)** | Goswami uses single synthetic-injection family + 2 internal measures, Borda aggregation | Six injection families + LCO at C in {2,4,8,16,32} + prediction-residual + ensemble-consensus, with anomaly-type-aware weighting |
| Selection-regret bound | **Yes** | Ma & Zhao 2023 showed IPMs are no better than random; no positive theory exists | Bound: pseudo-anomaly AUC ε-covers real AUC under (a) manifold-margin and (b) pseudo-anomaly diversity conditions |
| Anomaly-type-aware selection | **Yes** | Existing selectors are anomaly-agnostic | Use cluster-holdout score distribution as a free diagnostic of likely anomaly type, weight sources accordingly |
| Failure-mode characterization | Partial | Some negative results | Stratified analysis: which (dataset diagnostic, anomaly type) combinations break the method |

### 1.3 Direct competitors (full porting required)

| # | Method | Citation | Core idea | Setting | Differentiation from MS-PAS |
|---|---|---|---|---|---|
| C1 | **Goswami et al.** | ICLR 2023 (Oral). arXiv 2210.01078 | Prediction error + centrality + injected synthetic anomalies, robust rank aggregation | Univariate + multivariate TS-AD, zero-label | Single synthetic-injection family; no LCO; no type-aware weighting; no theory. **We strictly contain their signal set.** |
| C2 | **MSAD / "Choose Wisely"** | Sylligardos, Boniol, Paparrizos, Trahanias, Palpanas. PVLDB 2023 | Supervised meta-model maps TS characteristics to best of 12 detectors on TSB-UAD | Univariate TS-AD, **meta-supervised** (needs labeled historical pool) | Different supervision regime. We do **single-dataset, zero-label**. Include as upper-reference baseline (they get to use labels we don't). |
| C3 | **N-1 Experts** | Le Clei et al. AutoML-Conf 2022 LBW | For each candidate, use other N-1 detector predictions as pseudo-ground-truth | Algorithm-agnostic, zero-label | Pure consensus, no clustering, no targeted perturbation. We aggregate diverse pseudo-anomaly sources, not just detector agreement. |
| C4 | **Idan (ECAI 2024)** | arXiv 2410.14579 | Collaborative-decision paradigm: detector agreements/disagreements as validation signal | Tabular-leaning, zero-label | Detector-consensus only, no manifold/normal-data structure use. |
| C5 | **IREOS / IREOS-extension** | Marques et al. TKDD 2020; extension TKDD 2024 | Per-point separability via max-margin kernel logistic regression | Static tabular outliers | Internal index, not multi-source. Historical anchor; we benchmark against it. |
| C6 | **ASOI** | Complex & Intelligent Systems 2025 | Anomaly Separation and Overlap Index on score distributions | Score-based, modality-agnostic | Single internal index; we are a multi-source meta-selector. |

**Required result**: MS-PAS type-aware beats every zero-label competitor (C1, C3, C4, C5, C6) by **≥3% absolute VUS-PR regret on average across TSB-UAD + UCR**, with paired-bootstrap p < 0.05. We do NOT need to beat C2 (MSAD) since it uses meta-labels we deliberately deny ourselves; instead, MSAD serves as the "upper-reference" showing the value of historical labels.

### 1.4 Out-of-scope competitors (justified non-porting)

- **MetaOD (NeurIPS 2021)** and **ELECT (ICDM 2022)**: tabular, meta-supervised. We cite and discuss but do not port to TS; MSAD covers the meta-supervised TS slot.
- **SWSA / Fung et al. (IEEE TAI 2025)**: image-only, diffusion-based pseudo-anomaly generation. The *philosophy* (synthetic anomalies for validation) is captured by our Source 2; the *visual perturbation generator* does not transfer. Discussed in Related Work.
- **ADGym (NeurIPS 2023 D&B)**: few-label / weakly supervised design-choice search. Different setting entirely (some labels available). Discussed, not benchmarked.
- **Bootstrap stability, normal-loss, random, default**: trivial baselines, included as lower-bound anchors.

---

## 2. Scope Decisions (Locked)

1. **Modality**: univariate and multivariate time-series anomaly detection only. Defer tabular, image, text.
2. **Primary metric**: VUS-PR (Liu & Paparrizos NeurIPS 2024). Secondary: AUC-PR, AUC-ROC, range-based F1 (Tatbul et al.) for legacy comparability. Point-adjusted F1 explicitly excluded per "Elephant in the Room".
3. **Primary practical metric**: selection regret in VUS-PR vs oracle, averaged across datasets.
4. **Anomaly-label policy**: labels are touched only in (a) oracle computation and (b) final selection-regret evaluation. The selection procedure never sees them.
5. **One paper**, one cohesive story. No "and also images" extension in v1.

---

## 3. Datasets (Locked List)

### 3.1 Real benchmarks

| Suite | Series count (use) | Source | Notes |
|---|---|---|---|
| **TSB-AD (primary)** | Curated 1,070 univariate + multivariate, stratified subset of ~300 | https://github.com/TheDatumOrg/TSB-AD | Liu & Paparrizos NeurIPS 2024 D&B. Ships 40 detectors and VUS-PR built-in. **Supersedes TSB-UAD as the modern standard.** Apache 2.0. |
| TSB-UAD (legacy comparability) | 200 series, stratified sample | https://github.com/TheDatumOrg/TSB-UAD | For comparison against published MSAD numbers (PVLDB 2023). |
| UCR Anomaly Archive | All 250 series | https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ | Single anomaly per series, clean labels. |
| SMD (Server Machine Dataset) | 28 entities, multivariate | https://github.com/NetManAIOps/OmniAnomaly | For multivariate validation. |
| MSL + SMAP (NASA) | 27 + 55 entities | https://github.com/khundman/telemanom | Known label issues; secondary. |
| SWaT / WADI | Optional | https://itrust.sutd.edu.sg/itrust-labs_datasets/ | Access via form; ~3-day turnaround. |

**Subset rationale**: TSB-UAD ships 12,686 series. Running 60 candidate models on all of them is ~750k model-fits. We stratify-sample 250 series across the 18 source datasets, preserving proportional representation. We document the seed and the exact series IDs.

### 3.2 Controlled synthetic suite (we construct)

Generated from clean normal segments (no real anomalies) drawn from `ECG5000` normal class, `gait`, sinusoidal mixtures, and ARMA processes.

| Anomaly family | Generation recipe | # series |
|---|---|---|
| Point spike | Multiply random point by k in {3,5,10} sigma | 30 |
| Level shift | Add c in {2,4,8} sigma over a window of 5-20% length | 30 |
| Trend change | Inject linear slope of varying steepness | 30 |
| Frequency change | Replace window with same-mean signal at different dominant freq | 30 |
| Contextual | Value normal globally, abnormal vs local seasonality (swap into wrong phase) | 30 |
| Held-out mode | Train on subset of modes, anomaly = unseen mode | 30 |

Total synthetic: **180 series, 6 anomaly types, controlled difficulty levels**. This is the ablation backbone for "which anomaly types does LCO predict?".

### 3.3 Total benchmark size

~ **250 + 250 + 110 + 180 = 790 series**. Multivariate subset: ~110 series. Realistic for ~3-month single-researcher compute budget.

---

## 4. Candidate Model Pool (Locked List)

Total = **15 algorithm families × ~4 hyperparameter variants ≈ 60 candidate models**. Implemented behind one unified `BaseAD` interface (fit, score, predict).

### 4.1 Classical / shallow (CPU, fast)

| # | Algorithm | Library | Hyperparam grid |
|---|---|---|---|
| 1 | Isolation Forest | sklearn.ensemble.IsolationForest | n_estimators in {50, 100, 200, 500} |
| 2 | Local Outlier Factor | sklearn.neighbors.LocalOutlierFactor | n_neighbors in {5, 10, 20, 50} |
| 3 | One-Class SVM | sklearn.svm.OneClassSVM | nu in {0.01, 0.05, 0.1, 0.2}, kernel=rbf |
| 4 | KNN distance | pyod.models.knn | k in {5, 10, 20, 50}, method="mean" |
| 5 | PCA reconstruction | pyod.models.pca | n_components in {5, 10, 25, 50} |
| 6 | HBOS | pyod.models.hbos | n_bins in {10, 20, 50, 100} |
| 7 | COPOD | pyod.models.copod | no hyperparams, single config |
| 8 | ECOD | pyod.models.ecod | no hyperparams, single config |
| 9 | Robust covariance | sklearn.covariance.EllipticEnvelope | contamination in {0.01, 0.05, 0.1} |

### 4.2 Time-series specific (CPU, moderate)

| # | Algorithm | Library | Hyperparam grid |
|---|---|---|---|
| 10 | Matrix Profile (STOMP) | stumpy | m in {32, 64, 128, 256} subseq length |
| 11 | Seasonal decomposition residual | statsmodels.seasonal_decompose | period in {12, 24, 168, auto} |
| 12 | ARIMA residual | statsmodels.tsa.ARIMA | order grid (1,0,1), (2,0,1), (2,1,2) |

### 4.3 Deep learning (GPU, RTX 2060 6GB; train sparingly)

| # | Algorithm | Library / our impl | Hyperparam grid |
|---|---|---|---|
| 13 | LSTM Autoencoder | our pytorch impl | hidden in {32, 64}, layers in {1, 2} |
| 14 | TCN Autoencoder | our pytorch impl | channels in {32, 64}, kernel in {3, 5} |
| 15 | USAD | https://github.com/manigalati/usad | latent_size in {20, 40}, default else |

Deep models only trained on the **stratified 200-series subset** (random sample weighted by source) to keep GPU time tractable. The remaining 590 series get classical-only candidates. We report this scope decision in the paper.

### 4.4 Implementation contract

```python
# src/autoad/models/base.py
class BaseAD:
    name: str
    hyperparams: dict
    def fit(self, X: np.ndarray) -> "BaseAD": ...   # X shape (T, D), normal only
    def score(self, X: np.ndarray) -> np.ndarray: ... # returns (T,), higher = more anomalous
    @classmethod
    def grid(cls) -> list[dict]: ...                  # hyperparam combos
```

All 60 candidates inherit this. Score normalization to [0, 1] via empirical CDF on training data so cross-fold scores are comparable.

---

## 5. Method: MS-PAS (Multi-Source Pseudo-Anomaly Selection)

### 5.1 Source 1: Leave-Cluster-Out (LCO) Validation

```
Input:  normal training series X_N, candidate model m_k
Output: pseudo-anomaly AUC s_LCO(m_k)

1. Segment X_N into windows of length w (per-dataset chosen via seasonality).
2. For each cluster algorithm A in {KMeans, GMM, AgglomerativeWard, HDBSCAN, KShape}:
3.   For each C in {2, 4, 8, 16, 32} (skip if dataset too small):
4.     Cluster windows -> labels.
5.     Filter clusters by size >= 30 windows, separation, density.
6.     For each cluster c (held out as pseudo-anomaly):
7.       Train m_k on X_N \ c.
8.       Score held-out cluster c (pseudo-anomalous) and held-out IID slice of training clusters (pseudo-normal).
9.       Compute AUC-PR on pseudo-labels.
10.    Aggregate AUCs across folds via difficulty-stratified mean (see 5.4).
11. Aggregate across (A, C) via Borda / mean rank.
```

**Feature space for clustering**: TSFresh-extracted summary features (lightweight subset: mean, std, skew, kurt, autocorr lags 1/2/5/10, spectral entropy, dominant frequency, trend slope). For latent-domain variant, replace with PCA-50 of an autoencoder bottleneck trained on the full normal set (encoder fixed across folds to avoid circularity).

**Why multi-granularity**: small C tests coarse mode-exclusion (likely meaningful); large C tests fine-grained boundaries (likely noisy). Aggregation across C is what makes ranking stable.

### 5.2 Source 2: Synthetic Perturbation Pseudo-Anomalies (six families)

For each model m_k, on normal validation slice X_val:

```
for family in {point_spike, level_shift, trend, freq, contextual, mask}:
    X_pert = inject(X_val, family, severity grid)
    Score X_val (negative) and X_pert (positive).
    Compute AUC-PR family-wise.
```

We deliberately make these **disjoint from the real anomalies** in TSB-UAD/UCR (no peeking). Severities span easy/medium/hard.

### 5.3 Source 3: Prediction-Residual Consistency

For forecasting-capable models (LSTM-AE, TCN-AE, ARIMA, Matrix Profile via left-right join), compute residual time series on a held-out normal slice. The model whose residual is **most stationary and lowest-entropy on normals** is internally preferred. Use ADF-test p-value + permutation entropy as the score. This is the IPM that Ma & Zhao found weak alone; we include it as one of many signals.

### 5.4 Difficulty Stratification (the unfairness fix)

For each LCO fold, compute model-independent difficulty diagnostics on the held-out cluster:

- d1 = mean nearest-cluster Wasserstein distance in feature space
- d2 = MMD (RBF) between held-out and training distributions
- d3 = silhouette score of held-out cluster
- d4 = baseline separability AUC of a 2-class logistic regression on TSFresh features (no AD model involved)

Fold difficulty bucket = quantile bin of d4 across all folds. Report and aggregate **per bucket**. Trivial folds (d4 > 0.95) and impossible folds (d4 < 0.55) are excluded from the main aggregation but reported separately.

### 5.5 Aggregation: anomaly-type-aware rank fusion

Two stages:

**Stage A**: within each source, aggregate fold-level AUCs to a per-model score, then convert to rank.

**Stage B**: combine source ranks. Three combiners, all reported:

1. **Borda** (mean of ranks). Baseline.
2. **Plackett-Luce MLE** on ranks per source. Standard rank aggregation.
3. **Type-aware weighting**: estimate the dataset's likely dominant anomaly type from normal-data diagnostics (cluster compactness, seasonality strength, KL between cluster pairs), then weight sources by a learned weight matrix W[type, source]. W is fit on the controlled synthetic suite, never on real-anomaly labels. This is the novel piece.

### 5.6 Output

`hat_m = argmax_k score_combined(m_k)`

Returned together with a **confidence score**: rank-correlation among sources. Low confidence implies fallback to default (Isolation Forest 100 estimators).

---

## 6. Theoretical Component

### 6.1 Goal

Prove: under stated conditions, `Regret(MS-PAS) <= f(epsilon, delta)` where epsilon is pseudo-anomaly diversity and delta is manifold margin.

### 6.2 Setup

- Normal data distribution P on manifold M ⊂ R^d.
- Anomaly distribution Q supported outside an epsilon-tube around M.
- Pseudo-anomaly mixture P_tilde drawn from `union of (M-tube ∩ {held-out modes}) and (M-perturbations of severity sigma)`.
- AD model m: scoring function s_m: R^d -> R, AUC(m) = P(s_m(q) > s_m(p) | p ~ P, q ~ Q).

### 6.3 Target statement (sketch)

> If P_tilde covers Q in symmetric KL within delta, and candidate models are L-Lipschitz, then for any m1, m2 in our pool with AUC(m1) - AUC(m2) > c*delta + L*epsilon, the ordering on AUC_tilde matches the ordering on AUC.

Proof strategy: standard Lipschitz + coverage argument. Likely 2-3 pages of clean exposition. Empirical verification via: vary synthetic anomaly distance to normal manifold, measure when ranking flips.

### 6.4 Honest scope

This is **not** PAC. It is a conditional, distribution-dependent bound. We are explicit. The empirical paper carries the contribution; the theorem provides intuition for *when* it holds.

---

## 7. Baselines (Implemented in This Repo)

Three categories: (i) lower-bound anchors, (ii) zero-label competitors we must beat, (iii) upper-reference (meta-supervised) showing the value of historical labels we deny ourselves, (iv) ablations of our own method.

### 7.1 Lower-bound anchors

| ID | Baseline | Description | Effort |
|---|---|---|---|
| B0 | Random | Uniform random selection from the pool | trivial |
| B1 | Default | Isolation Forest n_estimators=100 (sklearn default) | trivial |
| B2 | Normal-loss | Reconstruction MSE on normal validation set (deep models only); selects model with **lowest** normal loss | easy |
| B3 | Bootstrap stability | Per-model inter-bootstrap score correlation; pick most stable | easy |

### 7.2 Zero-label competitors (we must beat these)

| ID | Baseline | Citation | Implementation effort | Compute cost on 790 series |
|---|---|---|---|---|
| C1 | **Goswami et al. full port** | ICLR 2023 Oral | medium-hard. Three signals + robust rank aggregation. Verify against any released code or published numbers on shared datasets. | ~5 h after oracle is computed |
| C3 | **N-1 Experts** | AutoML-Conf 2022 LBW | medium. Operates on already-computed candidate scores. Implement consensus pseudo-labeling + averaging. | <1 h (reuses oracle scores) |
| C4 | **Idan ECAI 2024** | arXiv 2410.14579 | medium-hard. Collaborative-decision formulation; adapt to windowed TS scores. Document adaptation choices. | ~3 h |
| C5 | **IREOS (fast variant)** | Marques et al. TKDD 2020 + extension TKDD 2024 | hard. Lift TS points to TSFresh windows; use the 2024 fast variant. The 2020 original is too slow for 790 series. | ~20 h |
| C6 | **ASOI** | Complex & Intelligent Systems 2025 | easy. Internal metric on already-computed scores; one-time formula. | <1 h |
| C7 | **Synthetic-injection only (Source 2 ablation, Goswami partial)** | derived | trivial (subset of our pipeline) | included in Source 2 cost |

### 7.3 Upper-reference (meta-supervised)

| ID | Baseline | Citation | Notes | Effort |
|---|---|---|---|---|
| C2 | **MSAD / Choose Wisely** | Sylligardos et al. PVLDB 2023 | Uses labeled meta-train. We retrain MSAD on 70% of TSB-UAD, evaluate on our 30% held-out series plus UCR (out-of-distribution). Code is public at https://github.com/boniolp/MSAD. **Not a fair zero-label peer**, but the most important reference point: the gap MS-PAS leaves on the table by refusing historical labels. | medium. Public code; reproduce their pipeline. |

### 7.4 Ablations of MS-PAS

| ID | Ablation | What it isolates |
|---|---|---|
| A1 | LCO only (Source 1) | Value of cluster-based pseudo-anomalies alone |
| A2 | Synthetic only (Source 2, all six families) | Value of perturbation pseudo-anomalies alone |
| A3 | Residual only (Source 3) | Value of internal IPMs alone (replicates Ma & Zhao negative result) |
| A4 | LCO + synthetic, no type weighting (Borda) | Effect of type-aware combiner |
| A5 | MS-PAS Plackett-Luce | Alternative aggregation |
| A6 | **MS-PAS type-aware** | **Headline method** |
| A7 | LCO single granularity (C=8 only) | Value of multi-granularity |
| A8 | LCO single cluster algo (KMeans only) | Value of cluster-algo ensembling |
| A9 | No difficulty stratification | Value of difficulty calibration |

### 7.5 Non-negotiables

We commit to full implementations of **C1 (Goswami), C2 (MSAD), C3 (N-1 Experts), C4 (Idan), C5 (IREOS fast), C6 (ASOI)**. These six are the comparison spine of the paper. Skipping any of them invites reviewer rejection on inadequate baselines.

Anomaly type per baseline (table for the paper, but commit to it now):

```
B0 random            : no anomaly assumption
B1 default           : no anomaly assumption
B2 normal-loss       : implicit (low normal loss => good detector); known weak
B3 bootstrap stab.   : detector stability under data resampling
C1 Goswami           : synthetic point injection + centrality + prediction error
C2 MSAD              : meta-supervised classification from TS characteristics
C3 N-1 Experts       : consensus pseudo-labels from other detectors
C4 Idan              : collaborative-decision agreement
C5 IREOS             : per-point separability via max-margin
C6 ASOI              : score distribution overlap index
A1-A9 MS-PAS variants: see ablation table
```

---

## 8. Evaluation Protocol

### 8.1 Three-stage discipline (audit-ready)

```
Stage 1 (offline, anomaly-label-free):
    train all 60 candidates on each dataset's normal split.
    compute pseudo-anomaly signals (Sources 1-3).
    compute selector outputs.

Stage 2 (oracle, uses anomaly labels):
    evaluate each candidate on the test split (normals + anomalies).
    compute VUS-PR, AUC-PR, AUC-ROC, range-F1.
    save to results DB. THIS IS THE ORACLE.

Stage 3 (regret computation):
    Regret(selector) = VUS-PR(oracle's best) - VUS-PR(selector's pick)
```

Stage 1 outputs are written before Stage 2 results are looked at by the researcher. Enforced via a gate: oracle file is in an encrypted zip, opened only after `01_stage_1_freeze.flag` exists.

### 8.2 Metrics reported

Per dataset, per selector:
- VUS-PR-Regret (primary)
- AUC-PR-Regret
- AUC-ROC-Regret
- Top-1 accuracy (did selector pick THE best model)
- Top-3 overlap with oracle's top-3
- Spearman rank correlation
- Kendall tau

Aggregated across datasets: mean, median, IQR, and significance via paired bootstrap (10k resamples).

### 8.3 Stratified analysis (the failure-mode tables)

For each (anomaly type, dataset diagnostic) bucket, report regret. Diagnostics:
- Series length
- Seasonality strength (Cleveland measure)
- Cluster count from BIC-optimal GMM on normal features
- Estimated anomaly contamination
- Anomaly type (one of point / contextual / collective / shift / unknown)

---

## 9. Phased Execution Plan

### Calendar assumption

Start: 2026-05-15. Target paper draft: 2026-08-15 (KDD 2027 Feb deadline allows refinement; ICDM 2026 if rush). Three months, single researcher.

### Phase 0 — Setup (Week 1, 2026-05-15 to 2026-05-21)

Deliverables:
- [ ] uv/pyproject toml; pinned env (Python 3.11, torch 2.x, sklearn, pyod, stumpy, tsfresh, hydra-core, mlflow)
- [ ] Repo structure as in Section 11
- [ ] `BaseAD` interface and three reference implementations (IForest, LOF, OCSVM) with passing unit tests
- [ ] CI: GitHub Actions running pytest + ruff on push
- [ ] MLflow tracking server local sqlite backend
- [ ] DVC or parquet-based artifact store under `runs/`

### Phase 1 — Data ingestion (Week 2, 05-22 to 05-28)

- [ ] Download scripts for TSB-UAD, UCR, SMD, MSL+SMAP, SWaT/WADI
- [ ] Stratified subset selector with frozen seed (`configs/subsets/v1.yaml`)
- [ ] Unified loader returning `(X_train_normal, X_test, y_test)` per series
- [ ] Synthetic generator producing the 180-series controlled suite
- [ ] Smoke test: load all 790 series in < 5 min, expected shapes, label sanity checks

### Phase 2 — Candidate pool and oracle (Weeks 3-4, 05-29 to 06-11)

- [ ] All 15 algorithm families behind `BaseAD`
- [ ] Hyperparameter grid → 60 candidates resolved
- [ ] Batch runner: 60 candidates × 790 series with checkpointing, restart-safe
- [ ] VUS-PR, AUC-PR, AUC-ROC, range-F1 evaluators (vendored from VUS repo)
- [ ] Oracle table in parquet: (series_id, model_id, metric, value)
- [ ] **Oracle frozen** under encrypted zip on 06-11. Hash recorded in repo.
- [ ] Sanity: spot-check 10 series against TSB-UAD published numbers, agreement within 0.02 AUC-PR

### Phase 3 — Pseudo-anomaly sources (Weeks 5-6, 06-12 to 06-25)

- [ ] Source 1 LCO with all 5 cluster algos × 5 C-values × 4 difficulty diagnostics
- [ ] Source 2 six synthetic perturbation families with severity grids
- [ ] Source 3 prediction-residual stationarity and entropy
- [ ] Per-source ranking outputs persisted
- [ ] Unit tests: trivial-fold rejection, impossible-fold rejection, score normalization correctness

### Phase 4 — Aggregation and selector (Week 7, 06-26 to 07-02)

- [ ] Borda combiner
- [ ] Plackett-Luce MLE combiner (use `choix` or hand-rolled)
- [ ] Type-aware combiner: train W on the synthetic 180-series suite, type-discrimination diagnostic from normal-data features
- [ ] Confidence score = inter-source rank correlation
- [ ] Sanity: on synthetic suite, type-aware should beat Borda by ≥ 2% regret

### Phase 5 — Baselines (Weeks 8-10, 07-03 to 07-23)

This phase is **expanded from 1 to 3 weeks** because we commit to six full competitor implementations (C1-C6) plus four anchors and nine ablations.

Week 8 (07-03 to 07-09): lower-bound anchors and ablation infrastructure
- [ ] B0 random, B1 default, B2 normal-loss, B3 bootstrap stability
- [ ] All MS-PAS ablations A1-A9 wired through the same selector harness
- [ ] Verification: ablations reproduce headline numbers from Phase 4

Week 9 (07-10 to 07-16): zero-label competitor ports (round 1)
- [ ] **C1 Goswami et al. ICLR 2023** full port. Verify against any released code; reach out to authors if needed. Match published numbers on at least 3 overlapping TSB-UAD datasets within 0.02 AUC-PR.
- [ ] **C3 N-1 Experts** (Le Clei et al. AutoML 2022 LBW). Algorithm-agnostic, reuses oracle scores.
- [ ] **C6 ASOI** (Complex & Intelligent Systems 2025). Internal metric on candidate scores.

Week 10 (07-17 to 07-23): zero-label competitor ports (round 2) and upper-reference
- [ ] **C4 Idan ECAI 2024** (arXiv 2410.14579). Collaborative-decision adaptation to windowed TS. Document adaptation.
- [ ] **C5 IREOS fast variant** (TKDD 2024 extension of Marques et al. TKDD 2020). Lift to TSFresh-window features. Skip if compute blows past 30 hours; in that case, run on a 200-series subset and report.
- [ ] **C2 MSAD / Choose Wisely** (Sylligardos et al. PVLDB 2023). Reproduce from public repo https://github.com/boniolp/MSAD. Retrain on 70% TSB-UAD, evaluate on our 30% held-out + UCR (OOD).
- [ ] Pre-flight check: every baseline produces a (series_id, selected_model_id) output in the standard format.

### Phase 6 — Main experiments and remaining ablations (Week 11, 07-24 to 07-30)

Most ablations are absorbed into Phase 5's harness work. This phase finalizes the headline tables.

- [ ] Main table: all selectors (B0-B3, C1-C6, A1-A9) × all datasets × all metrics
- [ ] Ablation: cluster algorithm (within Source 1)
- [ ] Ablation: cluster granularity C
- [ ] Ablation: representation (data vs latent for LCO)
- [ ] Ablation: candidate pool size (10, 20, 40, 60 candidates)
- [ ] Failure-mode tables stratified by anomaly type and series diagnostics
- [ ] Paired-bootstrap significance vs each of C1, C3, C4, C5, C6

### Phase 7 — Theory (Week 12, 07-31 to 08-06)

- [ ] Formal statement and proof of selection-regret bound
- [ ] Empirical verification on controlled synthetic suite (vary manifold margin, vary pseudo-anomaly diversity, plot regret vs bound)
- [ ] Honest discussion of bound tightness

### Phase 8 — Paper draft (Weeks 13-14, 08-07 to 08-20)

- [ ] 9-page main + appendix
- [ ] All tables and figures regenerated from `runs/` via `make figures`
- [ ] Co-author / advisor review pass
- [ ] Submit to chosen venue (or NeurIPS 2026 TS Workshop if Sep deadline targeted)

### Phase 9 — Buffer (08-21 to 09-15)

For inevitable slippage in baseline ports (Phase 5 is the highest-risk phase), additional ablations reviewers will demand, and writing polish. Target draft-frozen by 09-15 for any KDD 2027 / ICDM 2026 / NeurIPS 2026 Workshop submission window.

---

## 10. Repository Structure (Target)

```
AutoAD/
├── README.md
├── implementation plan.md          # this document
├── plan/
│   └── anomaly_detection_model_selection_plan.md
├── pyproject.toml                  # uv-managed
├── uv.lock
├── .pre-commit-config.yaml         # ruff + black + mypy
├── .github/workflows/ci.yml
├── configs/
│   ├── datasets/{tsb_uad.yaml, ucr.yaml, smd.yaml, ...}
│   ├── models/{iforest.yaml, lof.yaml, lstm_ae.yaml, ...}
│   ├── subsets/v1.yaml             # frozen subset IDs and seed
│   └── experiments/{e01_oracle.yaml, e02_lco.yaml, ...}
├── data/                           # gitignored
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/autoad/
│   ├── __init__.py
│   ├── data/
│   │   ├── loaders.py
│   │   ├── synthetic.py            # controlled suite generators
│   │   └── features.py             # TSFresh subset
│   ├── models/
│   │   ├── base.py                 # BaseAD interface
│   │   ├── classical.py            # IForest, LOF, OCSVM, KNN, PCA, HBOS, COPOD, ECOD, EllEnv
│   │   ├── ts.py                   # MatrixProfile, SeasonalResidual, ARIMA
│   │   └── deep.py                 # LSTM-AE, TCN-AE, USAD
│   ├── pseudo/
│   │   ├── cluster_holdout.py      # Source 1 (LCO)
│   │   ├── synthetic_perturb.py    # Source 2 (six families)
│   │   ├── residual.py             # Source 3
│   │   └── difficulty.py           # cluster diagnostics
│   ├── selection/
│   │   ├── aggregate.py            # Borda, Plackett-Luce
│   │   ├── type_aware.py           # type-discrim + weighted combiner
│   │   └── confidence.py
│   ├── eval/
│   │   ├── vus.py                  # vendored
│   │   ├── metrics.py              # AUC-PR, AUC-ROC, range-F1
│   │   └── regret.py
│   ├── theory/
│   │   └── bound.py                # numerical bound verification
│   └── utils/
│       ├── tracking.py             # MLflow wrappers
│       └── io.py
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_build_subset.py
│   ├── 03_train_oracle.py
│   ├── 04_freeze_oracle.py         # encrypts + hashes
│   ├── 05_run_pseudo_sources.py
│   ├── 06_run_selectors.py
│   ├── 07_run_baselines.py
│   ├── 08_compute_regret.py
│   ├── 09_make_tables.py
│   └── 10_make_figures.py
├── notebooks/
│   ├── eda_datasets.ipynb
│   ├── lco_walkthrough.ipynb
│   └── failure_mode_analysis.ipynb
├── tests/
│   ├── test_base_ad.py
│   ├── test_loaders.py
│   ├── test_pseudo_lco.py
│   ├── test_pseudo_synthetic.py
│   ├── test_aggregate.py
│   └── test_metrics.py
├── runs/                           # gitignored, MLflow + parquet
└── paper/
    ├── main.tex
    ├── refs.bib
    ├── figures/
    └── tables/
```

---

## 10.5 Artifact Reuse (decided 2026-05-14)

Critical finding: most of the infrastructure we need already exists. **We build by integration, not from scratch.**

### Reuse (clone, depend on, do not rewrite)

| Artifact | Repo | Role |
|---|---|---|
| **TSB-AD** (Liu & Paparrizos, NeurIPS 2024) | https://github.com/TheDatumOrg/TSB-AD | Primary benchmark + 40 detectors + VUS-PR. Supersedes TSB-UAD as the modern standard. Apache 2.0. |
| **TSB-AutoAD** | https://github.com/TheDatumOrg/TSB-AutoAD | AD model selection framework. Check overlap before building; may already implement N-1 Experts / IPM baselines. |
| **tsad-model-selection** (Goswami ICLR 2023) | https://github.com/mononitogoswami/tsad-model-selection | C1 baseline lives here. Apache 2.0. Reuse synthetic-injection module; verify reimpl against their numbers. |
| **MSAD** (Sylligardos PVLDB 2023) | https://github.com/boniolp/MSAD | C2 upper-reference. Pretrained weights shipped via Google Drive. MIT. |
| **PyOD** | https://github.com/yzhao062/pyod | IForest, LOF, OCSVM, KNN, PCA, HBOS, COPOD, ECOD, AutoEncoder, VAE-AD (10 of our 15 families). BSD-2. |
| **STUMPY** | https://github.com/TDAmeritrade/stumpy | Matrix Profile. BSD-3. |
| **Darts** | https://github.com/unit8co/darts | LSTM-AE, TCN-AE, forecasting residuals (replaces our hand-rolled deep impls). Apache 2.0. |
| **USAD** | https://github.com/manigalati/usad | Reference implementation. |
| **tsfresh** | https://github.com/blue-yonder/tsfresh | TSFresh feature panel. MIT. |
| **tslearn** | https://github.com/tslearn-team/tslearn | KShape, DTW K-Means. BSD-2. |
| **hdbscan, sklearn.cluster** | standard | KMeans, GMM, Ward, HDBSCAN. |
| **POT** | https://github.com/PythonOT/POT | Wasserstein-1, MMD. MIT. |
| **choix** | https://github.com/lucasmaystre/choix | Plackett-Luce MLE. MIT. |
| **TS2Vec** | https://github.com/zhihanyue/ts2vec | Contrastive encoder for **latent-domain LCO** (Source 1 latent variant). MIT. |
| **PRTS** | https://github.com/CompML/PRTS | Range-based F1 (Tatbul NeurIPS 2018). |
| **MetaOD, ELECT** | https://github.com/yzhao062/MetaOD, .../ELECT | Reference for meta-features (not direct baseline; out of scope per Sec 1.4). |
| **TimeEval** (optional) | https://github.com/HPI-Information-Systems/TimeEval | Dataset loaders for 700+ series. Heavy (Docker); use loaders only if TSB-AD coverage is insufficient. |

### Build (no reusable code exists)

| Component | Reason | Estimated effort |
|---|---|---|
| **LCO (Source 1)** | Novel; no precedent | core contribution, ~2 weeks |
| **Type-aware combiner** | Novel | ~3 days |
| **N-1 Experts port** (C3) | Paper only, no code released by Oracle Labs | ~3 days |
| **Idan ECAI 2024 port** (C4) | No public code | ~4 days |
| **ASOI port** (C6) | No public code, formula only | ~2 days |
| **IREOS Python port** (C5) | Repo exists but Java only (https://github.com/homarques/ireos-extension) | ~5 days (NumPy/Numba rewrite of ~600 LOC) |
| **Difficulty-stratified fold harness** | Specific to our protocol | ~3 days |
| **Stage 1/2/3 oracle-freeze gating** | Specific to our protocol | ~1 day |

### Net effect on schedule

Phase 1 (data ingestion) compresses from 1 week to 3 days (TSB-AD's loaders eliminate most work). Phase 2 (oracle) compresses from 2 weeks to 1 week (TSB-AD ships the 40 detectors; we add ~20 candidates from PyOD + Darts to reach 60). The freed time gets reinvested into Phase 5 baseline ports (now even more important since we have to build 4 of 6 from scratch).

**Revised compute estimate**: ~80 CPU-hours (down from 405) for the main pipeline because TSB-AD's detectors are optimized and many are vectorized; LCO inner loop remains the dominant cost.

## 11. Compute and Engineering

### 11.1 Hardware reality

- Local: RTX 2060 (6GB), Windows 11, Python 3.11 / 3.14
- GPU constraint: one task at a time, never `&`, always `run_in_background=true` (per CLAUDE.md)

### 11.2 Compute estimates

| Workload | Per-unit cost | Total | Hardware |
|---|---|---|---|
| Classical models × 60 × 790 series | ~5 s avg | ~66 h | CPU, parallel ~8h |
| Deep models × 12 variants × 200 series | ~3 min avg | ~120 h | RTX 2060 |
| Source 1 LCO × 60 × 790 × 5 cluster algos × 5 C × ~10 folds | ~1 s per fit | ~165 h | CPU, parallel ~20h |
| Source 2 synthetic × 60 × 790 × 6 families | ~2 s scoring | ~16 h | CPU |
| Source 3 residual × 12 deep variants × 790 | trivial | ~1 h | CPU |
| Baselines C1 Goswami | reuses oracle | ~5 h | CPU |
| Baseline C3 N-1 Experts | reuses oracle | <1 h | CPU |
| Baseline C4 Idan | reuses oracle | ~3 h | CPU |
| Baseline C5 IREOS fast | per-series max-margin | ~20 h (or 200-series subset) | CPU |
| Baseline C6 ASOI | reuses oracle | <1 h | CPU |
| Baseline C2 MSAD meta-train + inference | small NN training | ~5 h | CPU/GPU |
| Aggregation, metrics, regret | trivial | ~2 h | CPU |

**Grand total**: ~405 CPU hours, ~120 GPU hours. With aggressive parallelism (multiprocessing 8 workers for classical), realistic wall clock: ~55 CPU hours + 120 GPU hours, **~2-3 weeks if running continuously**. Baseline ports add ~35 hours compute (negligible vs engineering effort).

### 11.3 Optimization strategy

1. Cache window features once per series.
2. Cache cluster assignments once per (series, algo, C).
3. LCO inner loop uses **classical models only** for evaluating the *training* on held-out clusters (deep models are too slow per fold). This is a documented scope decision: LCO ranks classical models richly, ranks deep models at coarser granularity.
4. Synthetic perturbation is a cheap inference-only signal for already-fit deep models.
5. Aggressive checkpointing: every (series, model) result is written to parquet immediately. Crashes resume.

### 11.4 Optional offload

If local compute slips beyond Week 6, offload deep-model training to Modal (per `gpu2modal` skill) or RunPod. Budget: $50 cap, deep model training only. Estimated cost: ~$15 for 120 GPU-hours on a T4.

---

## 12. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LCO rankings unstable across cluster algos | Medium | High | Aggregate across algos and Cs; report stability ablation; the instability is itself a finding |
| Type-aware combiner overfits the 180-series synthetic suite | Medium | High | Cross-validate W on synthetic; test on held-out real datasets; report gap honestly |
| Goswami-2023 reimplementation off | Medium | High | Reach out to authors for any released code; verify on overlapping datasets where they publish numbers |
| Six baseline ports slip into Phase 6 | High | High | Phase 5 already expanded to 3 weeks; if C5 IREOS blows past 30 h compute, drop to 200-series subset; if C2 MSAD reproduction fails, fall back to citing their published numbers on TSB-UAD overlap |
| Idan ECAI 2024 adaptation contested | Medium | Medium | Document adaptation choices, release the adapted code, invite the author to validate (pre-submission contact is acceptable practice) |
| IREOS fast variant insufficiently fast | Medium | Medium | TKDD 2024 extension claims ~100x speedup; if still too slow, run on 200 series and report scaling separately |
| MSAD meta-train overfits our subset | Low | Low | Use their published splits where possible; document any deviation |
| Theory bound is trivial / vacuous | Medium | Medium | Even a non-vacuous distributional bound is publishable; if vacuous, drop to "empirical conditions" section |
| Compute overruns | Medium | Medium | Phase 6 has explicit fallback to 400-series subset; Modal offload |
| VUS-PR computation slow | Low | Low | Vendored implementation, cache per series |
| Anomaly label leakage | Low | Critical | Encrypted oracle file, explicit Stage 1 freeze flag |
| Reviewers ask for image/tabular extension | High | Low | Out of scope for v1; commit to follow-up in discussion |

---

## 13. Success Criteria (Quantitative Gates)

The project is a top-venue paper if **all** of (G1, G2, G3) and **most** of (G4-G8) hold.

| Gate | Description | Threshold |
|---|---|---|
| G1 | Headline regret of MS-PAS type-aware on TSB-UAD + UCR | mean VUS-PR regret **< 0.05 absolute** |
| G2 | Beat every zero-label competitor (C1, C3, C4, C5, C6) | **≥ 0.03 absolute** VUS-PR regret reduction vs each, paired-bootstrap p < 0.05 |
| G3 | Top-1 / Top-3 selection accuracy | Top-1 **≥ 30%**, Top-3 overlap **≥ 60%** out of 60 candidates |
| G4 | Mean Spearman vs oracle | **≥ 0.50** averaged across datasets |
| G5 | Gap to upper-reference (C2 MSAD) | report honestly; ideal **≤ 0.03 absolute regret** behind MSAD despite using no historical labels |
| G6 | Failure-mode characterization | clear (anomaly type × diagnostic) cells where method fails, explained from theory |
| G7 | Theory bound | at least one non-vacuous bound with empirical verification plot |
| G8 | Reproducibility | one-command reproduction; oracle hashes published; deterministic seeds |

**Decision tree**:
- G1 + G2 + G3 hold: KDD/NeurIPS submission, strong story.
- G2 partially holds (beat 3-4 of 5 zero-label competitors): KDD submission with cautious framing ("when MS-PAS wins and when it doesn't").
- G2 fails entirely (beat 0-2 competitors): pivot to "negative result + diagnostic framework" paper for a workshop (e.g., NeurIPS TS Workshop) and revisit method design.

The expanded baseline set (six full competitor ports) means **G2 is the highest-risk gate**. We invest disproportionate engineering effort in Phase 5 to make sure the ports are faithful, then either we beat them and have a clean paper or we don't and pivot honestly.

---

## 14. Paper Plan

### 14.1 Working title

**Leave-Cluster-Out: Multi-Source Pseudo-Anomaly Validation for Time-Series Anomaly Detection Model Selection**

### 14.2 Target sections and figures

1. Introduction (1 page)
2. Related work (0.75 page, table)
3. Problem setup and notation (0.5 page)
4. Method: MS-PAS (2.5 pages)
   - Fig 1: pipeline overview
   - Fig 2: LCO illustration
5. Theory (1.5 pages)
   - Fig 3: bound verification on synthetic
6. Experiments (3 pages)
   - Table 1: main results, all selectors × all benchmarks, VUS-PR regret
   - Table 2: top-k and rank correlation
   - Fig 4: per-anomaly-type performance
   - Table 3: ablations (sources, granularity, representation)
   - Fig 5: failure-mode heatmap
7. Discussion and limitations (0.5 page)
8. Appendix: dataset details, hyperparameter grids, additional ablations, theory proofs

### 14.3 Venue strategy

- Primary: **KDD 2027** (Feb 2027 deadline). Time-series and ML systems fit.
- Backup: **ICDM 2026** (June 2026 deadline) if results land by mid-May.
- Stretch: **NeurIPS 2027** (May 2027 deadline) with stronger theory.
- Workshop pre-submission: NeurIPS 2026 Time-Series Workshop (likely Sep 2026) for early feedback.

### 14.4 Reproducibility commitments

- Public release of code, configs, frozen subset IDs, oracle hashes
- All scripts deterministic with seeds
- One-command reproduction of every paper number from `runs/` and `paper/`

---

## 15. Memory and Documentation

After Phase 0, save the following memory entries:
- Project memory: "AutoAD paper aims for KDD/ICDM/NeurIPS 2026-2027 with MS-PAS multi-source pseudo-anomaly selection. Headline target: beat Goswami-2023 ICLR by 3% VUS-PR regret."
- Reference memory: TSB-UAD repo, UCR archive URL, VUS-PR implementation source
- Feedback memory: any user steer on scope (will update after first review)

---

## 16. Immediate Next Steps (this week)

In order:

1. User review and sign-off on this plan (especially Section 1 framing and Section 13 success gates).
2. Decide: solo project or invite a collaborator (theory or compute help).
3. Phase 0 execution starts: `pyproject.toml`, repo skeleton, three reference models, CI.
4. Phase 1 dataset download scripts.

I will not proceed past sign-off without explicit go-ahead.
