# Research Plan: Unsupervised Model Selection for Anomaly Detection via Cluster-Based Pseudo-Anomaly Validation

## 1. Executive Summary

Anomaly detection (AD) is commonly deployed in settings where labeled anomaly data are unavailable, scarce, incomplete, or unrepresentative. This creates a central methodological problem: many AD models can be trained using only normal data, but selecting the best model usually requires labeled anomalies. In realistic AD settings, using labeled anomalies for model selection is often impossible or scientifically questionable because the available anomalies may not represent the future anomaly distribution.

This project proposes and evaluates a family of **unsupervised model-selection protocols** for anomaly detection. The core idea is to use only normal data to construct pseudo-anomaly validation tasks. Normal data are clustered into subdomains, some clusters are held out during training, and anomaly detection models are evaluated by their ability to assign higher anomaly scores to held-out clusters than to in-distribution validation samples. The resulting pseudo-anomaly performance is used to rank candidate AD models.

Two variants are studied:

1. **Data-domain clustering**: clusters are formed using raw data or engineered domain features.
2. **Latent-domain clustering**: clusters are formed in a learned representation space, such as the latent space of an autoencoder, VAE, contrastive encoder, or forecasting encoder.

The scientific goal is not merely to propose another validation heuristic, but to study when and why such pseudo-anomaly rankings correlate with real anomaly performance. The work will compare cluster-based rankings against oracle rankings obtained using real anomaly labels reserved strictly for final analysis.

---

## 2. Motivation and Problem Statement

### 2.1 The Model-Selection Problem in Anomaly Detection

In standard supervised learning, model selection is performed using labeled validation data. In anomaly detection, this assumption usually fails. The training data often contain mostly normal samples, while labeled anomaly data are absent, incomplete, or too rare to support reliable validation.

This creates a paradox:

- We can train many anomaly detection models on normal data.
- But we cannot reliably choose between them without anomalies.
- If we had enough representative anomalies for validation, the problem would partly become supervised classification.

Therefore, anomaly detection requires a distinct model-selection methodology that does not depend on labeled anomalies.

### 2.2 Central Research Question

**Can we rank and select anomaly detection models using only normal data by constructing cluster-based pseudo-anomaly validation tasks, and under what conditions does this ranking predict performance on real anomalies?**

### 2.3 Practical Importance

A reliable unsupervised model-selection protocol would be useful in domains such as:

- Industrial monitoring
- Medical signal analysis
- Cybersecurity
- Predictive maintenance
- Financial fraud monitoring
- Sensor networks
- Operational time-series monitoring

In these settings, anomaly labels are often unavailable during development, and even when some anomalies exist, they may not represent future failures or attacks.

---

## 3. Initial Scientific Criticism of the Idea

The proposed idea is promising, but several issues must be addressed to make it suitable for a high-quality scientific paper.

### 3.1 Main Conceptual Weakness: Pseudo-Anomalies May Not Resemble Real Anomalies

Holding out a normal cluster creates a distribution-shift task, not necessarily an anomaly-detection task. A model that detects held-out normal subdomains may not detect true anomalies. For example, real anomalies may be local corruptions, rare spikes, abnormal temporal ordering, semantic violations, or subtle contextual deviations rather than entire missing modes.

**Required refinement:**
The paper should explicitly distinguish between:

- **Mode-exclusion anomalies**: samples from unseen normal modes.
- **Local perturbation anomalies**: small abnormal deviations from normal samples.
- **Structural anomalies**: abnormal temporal or spatial relationships.
- **Contextual anomalies**: values that are normal globally but abnormal in context.
- **Semantic anomalies**: violations of higher-level meaning.

The proposed method is most naturally aligned with mode-exclusion and distribution-shift anomalies. The paper should test whether it generalizes beyond them.

### 3.2 Risk of Circularity in Latent Clustering

If the latent representation is learned by a specific model family, the selection protocol may unfairly favor models with similar inductive biases. For example, VAE-based clustering may favor reconstruction-based anomaly detectors.

**Required refinement:**
Use multiple representation families for latent clustering and report whether model rankings are representation-dependent. Possible encoders include:

- Autoencoder embeddings
- VAE embeddings
- Contrastive embeddings
- Forecasting-model hidden states
- Self-supervised time-series embeddings
- Classical feature embeddings

The paper should include an ablation study showing whether rankings are stable across representation choices.

### 3.3 Cluster Quality Can Dominate the Result

Bad clustering can produce meaningless pseudo-anomaly tasks. Very distant clusters may create trivial tasks, while overlapping clusters may create impossible tasks. Imbalanced clusters may bias evaluation metrics.

**Required refinement:**
The protocol must include cluster diagnostics and difficulty calibration. Each held-out cluster fold should be characterized by:

- Cluster size
- Intra-cluster compactness
- Distance to nearest training cluster
- Separation from training clusters
- Baseline separability using a simple classifier
- Score distribution overlap between held-out and training clusters

The paper should report results stratified by pseudo-anomaly difficulty.

### 3.4 Model Selection Must Be Separated from Final Evaluation

A serious paper must avoid any leakage from real anomaly labels into the proposed selection procedure. Real anomaly labels can be used only for final evaluation of the proposed method.

**Required refinement:**
The experimental protocol should clearly define three stages:

1. **Candidate training** using normal training data.
2. **Unsupervised model selection** using only normal data and cluster-based pseudo-anomalies.
3. **Final evaluation** using real anomaly labels, only to assess whether the unsupervised ranking was correct.

### 3.5 Ranking Correlation Alone Is Not Enough

A method can achieve moderate rank correlation but still select a poor model. The most important practical question is selection regret.

**Required refinement:**
The main evaluation metric should include:

- Selection regret relative to the oracle model
- Top-1 model-selection accuracy
- Top-k overlap with oracle ranking
- Spearman correlation
- Kendall tau
- Difference in AUROC/AUPRC between selected and oracle-selected models

The primary practical metric should be:

> If the proposed method selects a model without using anomaly labels, how much performance is lost compared with the model that would have been selected using real anomaly labels?

### 3.6 Need for Strong Baselines

The paper must compare against existing unsupervised model-selection heuristics. Otherwise, reviewers may see the method as an isolated heuristic.

**Required refinement:**
Compare against baselines such as:

- Random model selection
- Average performance across synthetic noise perturbations
- Reconstruction error on normal validation data
- Stability under bootstrap resampling
- Self-supervised validation through data corruptions
- Model agreement / ensemble consensus
- Complexity-based model selection
- Validation likelihood on normal data, where applicable
- Hyperparameter default selection from common libraries

### 3.7 Need to Avoid Overclaiming

The proposed method cannot guarantee that the best pseudo-anomaly detector is the best real anomaly detector. The paper should not claim general unsupervised validation is solved.

**Required refinement:**
Frame the contribution as an empirical and methodological study of **normal-data-only model selection** under controlled pseudo-anomaly protocols, with explicit conditions under which the method succeeds or fails.

---

## 4. Refined Scientific Claim

A strong version of the claim would be:

> We investigate whether cluster-based pseudo-anomaly validation can serve as a normal-data-only model-selection protocol for anomaly detection. By holding out normal subdomains as pseudo-anomalies, we estimate each model's ability to define selective boundaries around learned regions of the normal data manifold. We evaluate whether this estimate predicts performance on real anomalies across datasets, model families, clustering strategies, and anomaly types.

A more cautious and scientifically defensible claim would be:

> Cluster-based pseudo-anomaly validation is not a universal substitute for labeled anomaly validation, but it can provide useful model-selection signal when the real anomaly distribution corresponds to out-of-manifold or mode-exclusion behavior. Its reliability depends on clustering quality, pseudo-anomaly difficulty, representation choice, and the type of real anomalies.

---

## 5. Proposed Contributions

The paper should be organized around the following contributions:

1. **A formal model-selection problem for unsupervised anomaly detection**
   - Define the setting where only normal data are available for model selection.
   - Distinguish training, selection, and final evaluation.

2. **Cluster-based pseudo-anomaly validation protocol**
   - Introduce leave-cluster-out validation for AD model ranking.
   - Define data-domain and latent-domain variants.

3. **Fairness-aware cluster evaluation**
   - Propose controls for cluster size, separation, and pseudo-anomaly difficulty.
   - Avoid trivial or impossible validation folds.

4. **Comprehensive empirical evaluation**
   - Compare pseudo-anomaly rankings with real anomaly rankings.
   - Evaluate across datasets, methods, hyperparameters, and anomaly types.

5. **Failure-mode analysis**
   - Identify when cluster-based validation is reliable and when it is misleading.

---

## 6. Formal Problem Setup

Let \(X_N = \{x_i\}_{i=1}^n\) be a set of normal training samples. Let \(\mathcal{M} = \{m_1, ..., m_K\}\) be a set of candidate anomaly detection models, including different algorithms and hyperparameter configurations.

Each model \(m_k\) is trained using normal data and produces an anomaly score:

\[
s_k(x) \in \mathbb{R}
\]

where larger values indicate greater abnormality.

The goal is to select a model:

\[
\hat{m} = \arg\max_{m_k \in \mathcal{M}} Q(m_k; X_N)
\]

where \(Q\) is a model-selection criterion computed using only normal data.

The true but unavailable target ranking is computed using a test set containing normal and anomaly samples:

\[
R_{real}(m_k) = Perf(m_k; X_{test}^{normal}, X_{test}^{anom})
\]

The research evaluates whether the proposed cluster-based criterion \(Q_{cluster}\) produces rankings that agree with \(R_{real}\).

---

## 7. Proposed Method

## 7.1 Data-Domain Cluster-Based Validation

### Step 1: Feature Construction

Construct feature vectors from normal data. For time series, possible feature families include:

- Statistical features: mean, variance, skewness, kurtosis
- Temporal features: trend, autocorrelation, change-point statistics
- Frequency features: spectral entropy, dominant frequencies, band power
- Shape features: dynamic time warping distances, motif features
- Window-level embeddings from simple encoders

### Step 2: Clustering

Cluster the normal samples into \(C\) clusters:

\[
X_N = C_1 \cup C_2 \cup ... \cup C_C
\]

Candidate clustering algorithms:

- k-means
- Gaussian mixture models
- hierarchical clustering
- spectral clustering
- HDBSCAN
- time-series k-means with DTW distance

### Step 3: Leave-Cluster-Out Validation

For each fold, select one or more clusters as pseudo-anomalies:

\[
X_{pseudo}^{anom} = C_j
\]

Train each AD model on the remaining clusters:

\[
X_{train}^{normal} = X_N \setminus C_j
\]

Evaluate the model's ability to separate held-out cluster samples from validation samples drawn from training clusters.

### Step 4: Ranking

Compute a pseudo-anomaly performance score for each model and aggregate across folds.

---

## 7.2 Latent-Domain Cluster-Based Validation

### Step 1: Representation Learning

Train a representation model on normal data:

\[
z_i = f_\theta(x_i)
\]

Possible representation models:

- Autoencoder
- Variational autoencoder
- Contrastive encoder
- Forecasting encoder
- Masked reconstruction encoder
- Transformer-based time-series encoder

### Step 2: Latent Clustering

Cluster the latent vectors \(z_i\) instead of raw inputs.

### Step 3: Leave-Cluster-Out Validation

Use the same leave-cluster-out procedure as in data-domain clustering.

### Step 4: Ranking

Rank candidate AD models by their ability to identify held-out latent clusters as pseudo-anomalous.

---

## 8. Fairness-Aware Cross-Validation Design

A naive leave-cluster-out protocol may be biased. The refined protocol should include the following controls.

### 8.1 Cluster Size Control

- Exclude clusters below a minimum size.
- Downsample large clusters.
- Weight each fold equally rather than weighting by number of samples.
- Report sensitivity to minimum cluster-size thresholds.

### 8.2 Difficulty Control

For each held-out cluster, estimate difficulty using model-independent diagnostics:

- Distance to nearest training cluster
- Silhouette score
- Density overlap
- Classifier separability between training clusters and held-out cluster
- Wasserstein or MMD distance between held-out and training distributions

Classify folds into:

- Easy pseudo-anomalies
- Medium pseudo-anomalies
- Hard pseudo-anomalies

Report ranking quality separately for each difficulty group.

### 8.3 Avoiding Trivial Tasks

Exclude or separately report folds where the held-out cluster is trivially separable. Otherwise, all models may perform well and ranking may become uninformative.

### 8.4 Avoiding Impossible Tasks

Exclude or separately report folds where the held-out cluster is nearly indistinguishable from the training clusters. Otherwise, ranking may mostly reflect noise.

### 8.5 Multiple Granularities

Repeat clustering with several values of \(C\). A reliable selection method should not depend on a single arbitrary cluster count.

Recommended cluster-count strategy:

- Use a range such as \(C \in \{4, 6, 8, 10, 15, 20\}\), adjusted for dataset size.
- Discard configurations that produce too many tiny clusters.
- Aggregate rankings across valid cluster granularities.

---

## 9. Candidate Anomaly Detection Models

The candidate pool should include diverse model families and hyperparameter variants.

### 9.1 Classical Methods

- Isolation Forest
- One-Class SVM
- Local Outlier Factor
- k-nearest-neighbor distance scoring
- Gaussian mixture density scoring
- Robust covariance / Elliptic Envelope

### 9.2 Time-Series Methods

- Forecasting-error models
- LSTM autoencoder
- Temporal convolutional autoencoder
- Transformer encoder reconstruction
- Matrix profile methods
- Seasonal decomposition residual scoring

### 9.3 Deep Representation Methods

- Autoencoder reconstruction error
- VAE reconstruction likelihood / ELBO-based score
- Deep SVDD
- Contrastive representation + distance-based scoring

### 9.4 Hyperparameter Variants

Each algorithm should be represented by multiple hyperparameter settings. This is important because the method should select not only between algorithms, but also between configurations of the same algorithm.

---

## 10. Datasets

### 10.1 Initial Focus: Time-Series Anomaly Detection

The first stage should focus on time-series datasets because many benchmarks provide normal training data and labeled anomalies for final evaluation.

Candidate benchmark families:

- Industrial sensor datasets
- Server-machine datasets
- NASA telemetry datasets
- Medical or physiological time-series datasets
- Synthetic controlled datasets with known anomaly types

### 10.2 Required Dataset Properties

Each dataset should provide:

- Sufficient normal data for clustering
- Labeled anomalies for final evaluation only
- Multiple anomaly types where possible
- Clear train/test separation
- Enough anomalies to compute stable oracle rankings

### 10.3 Controlled Synthetic Datasets

In addition to real benchmarks, controlled synthetic datasets should be included to test when the method is expected to work.

Synthetic anomaly types should include:

- Held-out-mode anomalies
- Point spikes
- Level shifts
- Trend changes
- Frequency changes
- Contextual anomalies
- Temporal-order anomalies

This allows testing whether cluster-based validation works only for mode-exclusion anomalies or also generalizes to other anomaly types.

---

## 11. Experimental Protocol

## 11.1 Stage A: Oracle Ranking Using Real Anomalies

For each dataset:

1. Train every candidate AD model on the normal training set.
2. Evaluate each model on a test set containing real normal and real anomalous samples.
3. Compute real anomaly performance metrics.
4. Produce the oracle ranking.

Real anomaly labels are used only here, for final analysis.

Recommended metrics:

- AUROC
- AUPRC
- Precision at top-k
- Recall at fixed false-positive rate
- F1 at selected operating points

For highly imbalanced anomaly detection, AUPRC and top-k precision may be more informative than AUROC.

---

## 11.2 Stage B: Unsupervised Ranking by Data-Domain Clustering

1. Compute data-domain features from normal training data.
2. Cluster the normal samples.
3. Run leave-cluster-out validation.
4. Train candidate models on selected clusters.
5. Evaluate on held-out clusters as pseudo-anomalies.
6. Aggregate fold-level results.
7. Produce data-domain pseudo-anomaly ranking.

---

## 11.3 Stage C: Unsupervised Ranking by Latent-Domain Clustering

1. Train representation models on normal training data.
2. Extract latent vectors.
3. Cluster the latent vectors.
4. Run leave-cluster-out validation.
5. Aggregate results.
6. Produce latent-domain pseudo-anomaly ranking.

---

## 11.4 Stage D: Ranking Evaluation

Compare each unsupervised ranking with the oracle ranking.

Use:

- Spearman rank correlation
- Kendall tau
- Top-1 agreement
- Top-3 overlap
- Normalized discounted cumulative gain over ranked models
- Selection regret

The most important metric should be selection regret:

\[
Regret = Perf(m^*_{oracle}) - Perf(\hat{m}_{unsup})
\]

where \(m^*_{oracle}\) is the best model according to real anomalies and \(\hat{m}_{unsup}\) is the model selected using only normal data.

---

## 12. Baselines for Model Selection

The proposed method should be compared against several model-selection baselines.

### 12.1 Random Selection

Randomly select a candidate model. This provides a lower-bound baseline.

### 12.2 Default Configuration

Use common default hyperparameters from standard libraries. This tests whether the proposed method improves over practical defaults.

### 12.3 Normal Validation Loss

For reconstruction or likelihood models, select the model with best validation performance on normal data. This is a common but flawed strategy because the model that reconstructs normal data best may not detect anomalies best.

### 12.4 Bootstrap Stability

Select models whose anomaly scores are stable under resampling of normal data.

### 12.5 Corruption-Based Pseudo-Anomalies

Generate pseudo-anomalies by corrupting normal samples, for example:

- Adding noise
- Masking segments
- Shuffling time windows
- Injecting spikes
- Warping time
- Swapping segments

Compare cluster-based pseudo-anomalies against corruption-based pseudo-anomalies.

### 12.6 Ensemble Agreement

Select models that agree most with an ensemble consensus over normal and perturbed samples.

---

## 13. Ablation Studies

A high-quality paper should include the following ablations.

### 13.1 Number of Clusters

Vary the number of clusters and measure ranking stability.

### 13.2 Clustering Algorithm

Compare k-means, GMM, hierarchical clustering, spectral clustering, and density-based clustering.

### 13.3 Representation Choice

Compare raw features, engineered features, autoencoder embeddings, VAE embeddings, and contrastive embeddings.

### 13.4 Latent Dimension

For VAE and autoencoder representations, vary the latent dimension.

### 13.5 Held-Out Cluster Strategy

Compare:

- Leave-one-cluster-out
- Leave-multiple-clusters-out
- Distance-controlled cluster holdout
- Size-balanced cluster holdout
- Difficulty-stratified cluster holdout

### 13.6 Candidate Pool Size

Test whether the method remains reliable as the number of candidate models and hyperparameter variants increases.

### 13.7 Anomaly Type

Analyze results by anomaly type where labels allow it.

This is crucial because cluster-based validation may work well for some anomaly types but fail for others.

---

## 14. Failure-Mode Analysis

The paper should devote significant space to cases where the method fails.

Possible failure modes:

1. **Cluster mismatch**
   - Clusters capture nuisance variation rather than anomaly-relevant structure.

2. **Pseudo-anomaly mismatch**
   - Held-out clusters differ from training clusters in ways unrelated to real anomalies.

3. **Representation bias**
   - Latent clustering favors models with similar representation assumptions.

4. **Trivial separability**
   - Held-out clusters are too easy to detect, so all models appear strong.

5. **Excessive overlap**
   - Held-out clusters are too similar to training clusters, so pseudo-anomaly evaluation is noisy.

6. **Model-family mismatch**
   - Some models detect global distribution shifts but fail on local anomalies.

7. **Score calibration effects**
   - Some models produce anomaly scores that are not comparable across folds without normalization.

8. **Overfitting to pseudo-anomaly protocol**
   - Hyperparameters selected by cluster validation may specialize to cluster separation rather than real anomaly detection.

---

## 15. Refined Experimental Design

### 15.1 Minimal Viable Paper

A focused first paper should avoid trying to cover every domain. A strong minimal version would include:

- 6-10 time-series anomaly detection datasets
- 8-12 AD model families/configurations
- Data-domain clustering
- Latent-domain clustering
- 3-4 model-selection baselines
- Ranking correlation and selection regret
- Ablations on cluster number, representation, and difficulty
- Failure-mode analysis by anomaly type

### 15.2 Recommended Scope Control

Do not start with time series, images, and text in the same paper. That would likely make the work broad but shallow.

Recommended structure:

- Paper 1: Time-series anomaly detection
- Paper 2: Extension to images/text, if Paper 1 produces strong evidence

### 15.3 Suggested Main Tables

#### Table 1: Dataset Summary

Include dataset size, number of normal samples, number of anomalies, anomaly types, train/test split, and domain.

#### Table 2: Candidate Models

List each AD model family, hyperparameter variants, and score definition.

#### Table 3: Ranking Agreement

Compare data-domain clustering, latent-domain clustering, and baselines using Spearman, Kendall, top-k overlap, and regret.

#### Table 4: Ablation Results

Show sensitivity to number of clusters, representation type, and clustering method.

#### Table 5: Failure-Mode Summary

Show where the method succeeds and fails by dataset and anomaly type.

---

## 16. Recommended Paper Title Options

1. **Selecting Anomaly Detection Models Without Anomalies: Cluster-Based Pseudo-Anomaly Validation**
2. **Unsupervised Model Selection for Anomaly Detection via Leave-Cluster-Out Validation**
3. **Can Normal Data Select Anomaly Detectors? A Study of Cluster-Based Pseudo-Anomaly Validation**
4. **Normal-Only Validation for Anomaly Detection Model Selection**
5. **Pseudo-Anomaly Cross-Validation for Unsupervised Anomaly Detection**

The third title is likely the strongest for a scientific paper because it frames the work as an empirical question rather than an overclaim.

---

## 17. Recommended Abstract Draft

Selecting anomaly detection models is difficult when labeled anomalies are unavailable, incomplete, or unrepresentative. Although many anomaly detection methods can be trained using only normal data, choosing among them usually requires labeled anomalies, creating a fundamental model-selection problem. We study cluster-based pseudo-anomaly validation as a normal-data-only model-selection protocol. The proposed approach clusters normal data, holds out selected clusters as pseudo-anomalies, trains candidate anomaly detectors on the remaining clusters, and ranks models by their ability to separate held-out clusters from in-distribution normal samples. We compare two variants: clustering in data-domain feature space and clustering in learned latent representations. Across multiple time-series anomaly detection datasets, candidate model families, clustering algorithms, and representation choices, we evaluate whether pseudo-anomaly rankings predict oracle rankings computed using real anomaly labels reserved only for final analysis. We measure rank correlation, top-k agreement, and selection regret relative to oracle model selection. The study further analyzes failure modes caused by cluster quality, pseudo-anomaly difficulty, representation bias, and anomaly-type mismatch. The results aim to clarify when normal-only model selection is reliable and when pseudo-anomaly validation provides misleading signals.

---

## 18. Refined Research Plan

### Phase 1: Literature Review and Positioning

Review work on:

- Unsupervised anomaly detection
- Model selection without labels
- Self-supervised validation
- Pseudo-anomaly generation
- Out-of-distribution detection validation
- Cluster-based cross-validation
- Time-series anomaly benchmarks

The literature review should position this work as a model-selection study, not only as another AD algorithm.

### Phase 2: Dataset and Candidate Model Preparation

- Select time-series benchmark datasets.
- Standardize preprocessing and train/test splits.
- Define candidate AD model pool.
- Include both algorithmic diversity and hyperparameter diversity.
- Implement a unified scoring interface.

### Phase 3: Oracle Evaluation

- Train all candidate models on normal training data.
- Evaluate them on real anomalies.
- Compute oracle rankings.
- Store results without using them during unsupervised selection.

### Phase 4: Data-Domain Pseudo-Anomaly Validation

- Extract data-domain features.
- Generate multiple clusterings.
- Run leave-cluster-out validation.
- Compute pseudo-anomaly rankings.
- Analyze ranking agreement with oracle rankings.

### Phase 5: Latent-Domain Pseudo-Anomaly Validation

- Train representation models.
- Cluster latent vectors.
- Run leave-cluster-out validation.
- Compare rankings with data-domain clustering and oracle rankings.

### Phase 6: Baselines and Ablations

- Compare against random selection, default configurations, normal validation loss, bootstrap stability, and corruption-based pseudo-anomalies.
- Run ablations on clustering algorithm, number of clusters, latent dimension, representation type, and fold difficulty.

### Phase 7: Failure-Mode and Interpretation Analysis

- Analyze disagreements between pseudo-anomaly rankings and oracle rankings.
- Study which anomaly types are well predicted by cluster-based validation.
- Identify dataset characteristics associated with success or failure.

---

## 19. Stronger Scientific Framing

The paper should avoid claiming that cluster-based pseudo-anomaly validation solves anomaly detection model selection in general. A stronger and more credible framing is:

> This work studies the conditions under which normal-data structure contains enough information to guide anomaly detection model selection.

This framing makes the work scientifically valuable even if the method does not always outperform baselines. Negative or mixed results can still be publishable if the analysis explains when normal-only validation works and when it fails.

---

## 20. Key Risks and Mitigations

| Risk | Why It Matters | Mitigation |
|---|---|---|
| Pseudo-anomalies do not match real anomalies | Method may select wrong models | Analyze by anomaly type; include controlled synthetic datasets |
| Clusters are arbitrary | Results may be unstable | Use multiple clustering algorithms and cluster counts |
| Latent representations bias results | Selection may favor similar models | Use multiple encoders and representation families |
| Ranking metrics look good but selected model is poor | Correlation may be misleading | Use selection regret as primary practical metric |
| Too many experimental dimensions | Paper becomes unfocused | Start with time-series only |
| Weak baselines | Reviewers may reject contribution | Include random, default, normal-loss, stability, and corruption baselines |
| Data leakage | Invalidates conclusions | Strictly separate unsupervised selection from final anomaly-label evaluation |

---

## 21. Final Recommended Version of the Research Objective

This research investigates normal-data-only model selection for anomaly detection. The main hypothesis is that the internal structure of normal data can provide useful validation signals for selecting anomaly detection models, even when labeled anomalies are unavailable. The proposed approach creates pseudo-anomaly tasks by clustering normal data and holding out selected clusters during training. Candidate anomaly detectors are ranked by their ability to distinguish held-out normal subdomains from the training distribution. The study compares data-domain and latent-domain clustering, evaluates ranking agreement with real anomaly performance, and analyzes the conditions under which cluster-based validation succeeds or fails. The expected contribution is a rigorous empirical framework for anomaly detection model selection under realistic no-anomaly-label constraints.
