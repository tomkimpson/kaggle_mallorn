# MALLORN Challenge: Progress Review

**Project**: Tidal Disruption Event Classification
**Competition**: [Kaggle MALLORN Astronomical Classification Challenge](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)
**Last Updated**: January 2026

---

## 1. Problem Statement

### Scientific Background

Tidal Disruption Events (TDEs) occur when a star passes close enough to a supermassive black hole to be torn apart by tidal forces. These events are scientifically valuable for studying black hole demographics and accretion physics, but remain rare—only ~100 TDEs have been observed to date.

The upcoming Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST) will discover millions of transient events, among which TDEs will be a small but important fraction. Automated classification methods are essential to identify TDEs from photometric data alone.

### Task Definition

- **Objective**: Binary classification — TDE (1) vs non-TDE (0)
- **Evaluation Metric**: F1 Score
- **Primary Challenge**: Severe class imbalance (~5% positive class)

---

## 2. Dataset Overview

### Data Statistics

| Split | Objects | TDEs | Non-TDEs | TDE Fraction | Total Observations |
|-------|---------|------|----------|--------------|-------------------|
| Train | 3,043 | 148 | 2,895 | 4.9% | 479,384 |
| Test | 7,135 | ? | ? | ? | 1,145,125 |

### Data Structure

Each astronomical object contains:

1. **Light Curves**: Time-series flux measurements in 6 LSST photometric bands
   - Bands: u, g, r, i, z, y (ultraviolet to near-infrared)
   - Variable-length sequences (irregular cadence)
   - Features per observation: MJD (time), flux, flux error

2. **Metadata**:
   - Redshift (Z): Distance/cosmological redshift
   - Extinction (EBV): Milky Way dust extinction coefficient

### Non-TDE Classes (Contaminants)

The training set includes various transient types that must be distinguished from TDEs:
- Active Galactic Nuclei (AGN) — dominant class
- Type Ia Supernovae (SN Ia)
- Core-collapse Supernovae (SN II, SN Ib/c)
- Superluminous Supernovae (SLSN-I, SLSN-II)
- Other transients

---

## 3. Approaches Implemented

### 3.1 Deep Learning Models

We implemented two neural network architectures with per-band encoders that process each photometric filter independently before fusion.

#### 3.1.1 LightCurveCNN

A 1D Convolutional Neural Network with:
- 4 convolutional layers per band (6 parallel encoders)
- Global average pooling for sequence aggregation
- Classification MLP with metadata fusion
- ~545K trainable parameters

#### 3.1.2 LightCurveTransformer

A Transformer-based architecture with:
- Self-attention encoder per band
- Cross-band attention for multi-wavelength fusion
- Positional encoding for temporal information
- ~175K trainable parameters

#### Deep Learning Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch Size | 64 |
| Optimizer | AdamW |
| Learning Rate | 1e-3 (CNN), 5e-4 (Transformer) |
| Loss Function | Focal Loss (α=0.75, γ=2.0) |
| Early Stopping | 20 epochs patience |
| Cross-Validation | 5-fold stratified |

### 3.2 Feature-Based Models

#### 3.2.1 ROCKET + LightGBM (Baseline)

ROCKET (Random Convolutional Kernel Transform) is a time series feature extraction method that:
- Applies thousands of random convolutional kernels with varying dilations
- Extracts max value and proportion of positive values per kernel
- Creates a fixed-length feature vector regardless of input length

**Pipeline**:
1. Interpolate irregular light curves to regular 100-point grid per band
2. Apply ROCKET transform (2,000 or 5,000 kernels per band)
3. Concatenate features across all 6 bands
4. Train LightGBM gradient boosting classifier

**LightGBM Configuration**:
| Parameter | Value |
|-----------|-------|
| Boosting Type | GBDT |
| Num Leaves | 31 |
| Learning Rate | 0.05 |
| Feature Fraction | 0.8 |
| Bagging Fraction | 0.8 |
| Early Stopping | 50 rounds |

#### 3.2.2 ROCKET + Domain Features (New)

Building on the baseline ROCKET approach, we added **138 physics-informed domain features** to capture astrophysically meaningful characteristics of TDEs.

**Domain Features (per band, 6 bands × 20 features = 120 features)**:
- **Peak characteristics**: peak_flux, peak_time, mean_flux, std_flux, median_flux
- **Statistical moments**: skewness, kurtosis
- **Temporal evolution**: rise_time, decay_rate, amplitude_ratio, duration
- **Signal quality**: above_baseline_frac, snr_mean, snr_max
- **TDE-specific**: decay_chi2, decay_alpha_dev (deviation from t^-5/3 power law)
- **Smoothness metrics**: n_inflection_points, max_acceleration, monotonic_ratio, roughness

**Cross-band features (18 features)**:
- Color indices: g-r, r-i, i-z, u-g
- Peak time differences between bands
- Total flux statistics
- Color evolution: g-r and u-g colors at peak, at +30 days, and evolution rate

**Combined feature space**: 60,000 (ROCKET) + 138 (domain) = 60,138 features

#### 3.2.3 ROCKET + Domain + Optuna Tuning (In Progress)

We are currently running Optuna hyperparameter optimization to find optimal LightGBM parameters for the domain-enhanced feature set.

**Optuna Search Space**:
- num_leaves: [15, 127]
- learning_rate: [0.01, 0.3]
- feature_fraction: [0.4, 1.0]
- bagging_fraction: [0.4, 1.0]
- min_child_samples: [5, 100]
- reg_alpha: [1e-8, 10.0]
- reg_lambda: [1e-8, 10.0]

**Early Results** (2 trials completed before timeout):
- Trial 0: F1 = 0.473
- Trial 1: F1 = **0.541** (significant improvement!)

**Current Status**: Job 8156808 running with 20 trials, 30-hour time limit.

---

## 4. Handling Class Imbalance

Given the severe class imbalance (5% TDEs), we employed multiple strategies:

1. **Focal Loss** (Deep Learning)
   - Down-weights well-classified examples
   - Focuses learning on hard negatives
   - Parameters: α=0.75, γ=2.0

2. **Class Weighting** (LightGBM)
   - `scale_pos_weight` = ratio of negatives to positives (~19.5)

3. **Threshold Optimization**
   - Find optimal classification threshold on validation set
   - Optimize for F1 rather than using default 0.5

4. **Stratified Cross-Validation**
   - Maintain class distribution across all folds

---

## 5. Data Preprocessing

1. **De-extinction Correction**
   - Apply Fitzpatrick99 dust law using EBV values
   - Correct observed fluxes for Milky Way extinction

2. **Feature Engineering**
   - Time delta from first observation
   - Signal-to-noise ratio (flux / flux_err)
   - Log-transformed flux
   - Cumulative observation time

3. **Normalization**
   - Per-object flux normalization (divide by max absolute value)

4. **Sequence Handling**
   - Fixed padding to 200 observations per band (deep learning)
   - Linear interpolation to 100-point grid (ROCKET)

---

## 6. Results

### 5-Fold Cross-Validation Performance

| Model | F1 Score | Precision | Recall | ROC-AUC | Threshold |
|-------|----------|-----------|--------|---------|-----------|
| **ROCKET + Domain + Optuna** | **~0.54** | - | - | - | - | *In progress* |
| **ROCKET + Domain (5K)** | **0.481 ± 0.097** | 0.436 | 0.595 | 0.915 | 0.16 |
| ROCKET + LightGBM (5K) | 0.458 ± 0.105 | 0.445 | 0.481 | 0.902 | 0.18 |
| ROCKET + LightGBM (2K) | 0.462 ± 0.112 | 0.431 | 0.540 | 0.890 | 0.16 |
| Transformer | 0.191 ± 0.033 | 0.145 | 0.352 | 0.732 | 0.66 |
| CNN | 0.166 ± 0.019 | 0.112 | 0.338 | 0.566 | 0.61 |

### Per-Fold Breakdown: ROCKET + Domain Features (5K)

| Fold | F1 | Precision | Recall | ROC-AUC |
|------|-----|-----------|--------|---------|
| 0 | 0.324 | 0.227 | 0.567 | 0.866 |
| 1 | 0.490 | 0.632 | 0.400 | 0.904 |
| 2 | 0.630 | 0.535 | 0.767 | 0.952 |
| 3 | 0.475 | 0.373 | 0.655 | 0.930 |
| 4 | 0.486 | 0.415 | 0.586 | 0.925 |

### Per-Fold Breakdown: ROCKET + LightGBM (5K, Baseline)

| Fold | F1 | Precision | Recall | ROC-AUC |
|------|-----|-----------|--------|---------|
| 0 | 0.296 | 0.333 | 0.267 | 0.828 |
| 1 | 0.414 | 0.429 | 0.400 | 0.893 |
| 2 | 0.600 | 0.600 | 0.600 | 0.953 |
| 3 | 0.540 | 0.500 | 0.586 | 0.920 |
| 4 | 0.438 | 0.364 | 0.552 | 0.917 |

### Submissions Generated

| Submission File | Model | CV F1 | Test Predictions |
|-----------------|-------|-------|------------------|
| `submission_rocket_domain_5k.csv` | ROCKET+Domain (5K) | 0.481 | - |
| `submission_rocket_lgbm_5k.csv` | ROCKET+LGBM (5K) | 0.458 | 317 TDEs |
| `submission_rocket_lgbm_2k.csv` | ROCKET+LGBM (2K) | 0.462 | 408 TDEs |
| `submission_transformer_ensemble.csv` | Transformer (5-fold) | 0.191 | - |
| `submission_baseline_cnn.csv` | CNN | 0.166 | - |

---

## 7. Key Findings

### 7.1 Feature-Based Methods Outperform Deep Learning

The ROCKET + LightGBM approach achieved **2.4x higher F1 scores** than deep learning models. Likely reasons:
- Small dataset size (3,043 objects) limits deep learning
- ROCKET's random kernels capture diverse temporal patterns
- Gradient boosting handles tabular features effectively

### 7.2 Domain Features Improve Performance

Adding 138 physics-informed features improved CV F1 from 0.458 to **0.481** (+5%):
- Higher recall (0.595 vs 0.481) - catching more TDEs
- Improved ROC-AUC (0.915 vs 0.902)
- Key features likely include color evolution and decay characteristics

### 7.3 Optuna Shows Strong Potential

Early Optuna trials suggest significant headroom for improvement:
- Trial 1 achieved F1 = **0.541** vs baseline 0.481
- Hyperparameter tuning appears critical for this feature-rich model
- Full 20-trial optimization in progress

### 7.4 High Variance Across Folds

F1 scores vary significantly across folds (0.30 to 0.63), indicating:
- TDE population heterogeneity
- Limited positive samples per fold (~30)
- Model sensitivity to specific TDE subtypes

### 7.5 Precision-Recall Trade-off

- Deep learning models have low precision (~11-14%) but moderate recall (~34-35%)
- ROCKET models achieve better balance (precision ~44%, recall ~48-60%)
- Domain features shift toward higher recall at some precision cost
- Optimal thresholds are well below 0.5 for ROCKET (~0.16-0.18)

---

## 8. Infrastructure

- **Compute**: SLURM cluster (OzSTAR) with CUDA GPUs
- **Framework**: PyTorch 2.0+, scikit-learn, LightGBM, Optuna
- **Training Time**:
  - ~1 hour for ROCKET+LGBM feature extraction
  - ~45 min per Optuna trial (5-fold CV with 60K features)
  - ~2 hours for deep learning (5-fold)
- **Task Tracking**: `bd` (beads) for issue management

---

## 9. Progress Timeline

| Date | Milestone | CV F1 |
|------|-----------|-------|
| Dec 22 | CNN baseline complete | 0.166 |
| Dec 22 | Transformer baseline complete | 0.191 |
| Dec 23 | ROCKET + LightGBM (5K) | 0.458 |
| Dec 24 | ROCKET + LightGBM (2K) | 0.462 |
| Dec 26 | ROCKET + Domain Features | 0.481 |
| Jan 5 | Optuna tuning started | ~0.54 (early) |

---

## 10. Next Steps

### Immediate (In Progress)
1. **Complete Optuna Tuning** (Job 8156808)
   - 20 trials with domain-enhanced features
   - Expected completion: ~15-20 hours

### Short Term
2. **Ensemble Stacking**
   - Combine ROCKET variants with optimized model
   - Weighted averaging of diverse models
   - Scripts ready: `scripts/ensemble_stack.py`

3. **XGBoost Comparison**
   - Alternative to LightGBM
   - Script ready: `scripts/train_rocket_xgboost.py`

### Medium Term
4. **Feature Selection**
   - Identify most predictive domain features
   - Reduce dimensionality for faster training

5. **Calibration**
   - Isotonic regression on predicted probabilities
   - Better threshold optimization

### Longer Term
6. **Semi-Supervised Learning**
   - Pseudo-labeling on high-confidence test predictions
   - Self-training iterations

7. **Data Augmentation**
   - Time warping
   - Noise injection
   - Synthetic TDE generation

---

## 11. Summary

| Approach | Status | Best CV F1 |
|----------|--------|------------|
| 1D CNN | Complete | 0.166 |
| Transformer | Complete | 0.191 |
| ROCKET + LightGBM (2K) | Complete | 0.462 |
| ROCKET + LightGBM (5K) | Complete | 0.458 |
| ROCKET + Domain Features | Complete | 0.481 |
| ROCKET + Domain + Optuna | **In Progress** | ~0.54 (early) |

**Current best approach**: ROCKET + Domain Features with Optuna tuning, showing early F1 of ~0.54.

The feature-based approach substantially outperforms deep learning on this dataset. Adding physics-informed domain features and hyperparameter optimization are proving to be effective strategies for pushing performance higher.

---

## 12. Files & Checkpoints

```
checkpoints/
├── cnn/                    # CNN model weights (5 folds)
├── transformer/            # Transformer weights (5 folds)
├── rocket/                 # Original ROCKET features
├── rocket_lgbm_2k/         # ROCKET 2K + LGBM
├── rocket_lgbm_5k/         # ROCKET 5K + LGBM
├── rocket_domain_5k/       # ROCKET 5K + Domain features
│   ├── train_features.npz  # Cached features (60,138 dims)
│   ├── cv_summary.yaml     # Cross-validation results
│   └── fold_*/             # Per-fold models
└── rocket_domain_optuna/   # Optuna-tuned model (in progress)
```

---

## References

- [MALLORN Dataset Paper](https://arxiv.org/abs/2512.04946)
- [ROCKET: Exceptionally fast and accurate time series classification](https://arxiv.org/abs/1910.13051)
- [TDE Review: Gezari 2021](https://arxiv.org/abs/2104.14580)
