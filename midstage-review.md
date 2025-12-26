# MALLORN Challenge: Mid-Stage Review

**Project**: Tidal Disruption Event Classification
**Competition**: [Kaggle MALLORN Astronomical Classification Challenge](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)
**Date**: December 2025

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

#### 3.2.1 ROCKET + LightGBM

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
| **ROCKET + LightGBM (5K)** | **0.458 ± 0.105** | 0.445 | 0.481 | 0.902 | 0.18 |
| **ROCKET + LightGBM (2K)** | **0.462 ± 0.112** | 0.431 | 0.540 | 0.890 | 0.16 |
| Transformer | 0.191 ± 0.033 | 0.145 | 0.352 | 0.732 | 0.66 |
| CNN | 0.166 ± 0.019 | 0.112 | 0.338 | 0.566 | 0.61 |

### Per-Fold Breakdown (Best Model: ROCKET + LightGBM 5K)

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
| `submission_rocket_lgbm_5k.csv` | ROCKET+LGBM (5K kernels) | 0.458 | 317 TDEs |
| `submission_rocket_lgbm_2k.csv` | ROCKET+LGBM (2K kernels) | 0.462 | 408 TDEs |
| `submission_transformer_ensemble.csv` | Transformer (5-fold) | 0.191 | - |
| `submission_baseline_cnn.csv` | CNN | 0.166 | - |

---

## 7. Key Findings

### 7.1 Feature-Based Methods Outperform Deep Learning

The ROCKET + LightGBM approach achieved **2.4x higher F1 scores** than deep learning models. Likely reasons:
- Small dataset size (3,043 objects) limits deep learning
- ROCKET's random kernels capture diverse temporal patterns
- Gradient boosting handles tabular features effectively

### 7.2 High Variance Across Folds

F1 scores vary significantly across folds (0.30 to 0.60), indicating:
- TDE population heterogeneity
- Limited positive samples per fold (~30)
- Model sensitivity to specific TDE subtypes

### 7.3 Precision-Recall Trade-off

- Deep learning models have low precision (~11-14%) but moderate recall (~34-35%)
- ROCKET models achieve better balance (precision ~44%, recall ~48-54%)
- Optimal thresholds are well below 0.5 for ROCKET (~0.16-0.18)

---

## 8. Infrastructure

- **Compute**: SLURM cluster with CUDA GPUs
- **Framework**: PyTorch 2.0+, scikit-learn, LightGBM
- **Training Time**: ~1 hour for ROCKET+LGBM, ~2 hours for deep learning (5-fold)

---

## 9. Potential Next Steps

1. **Ensemble Methods**
   - Stack predictions from multiple models
   - Weighted average of ROCKET variants

2. **Additional Features**
   - Hand-crafted astronomical features (rise time, decay rate, colors)
   - Gaussian Process interpolation for smoother light curves

3. **Hyperparameter Optimization**
   - Optuna-based tuning for LightGBM
   - Architecture search for neural networks

4. **Semi-Supervised Learning**
   - Pseudo-labeling on high-confidence test predictions
   - Self-training iterations

5. **Data Augmentation**
   - Time warping
   - Noise injection
   - Synthetic TDE generation

---

## 10. Summary

We have implemented and evaluated four approaches for TDE classification:

| Approach | Status | Best CV F1 |
|----------|--------|------------|
| 1D CNN | Complete | 0.166 |
| Transformer | Complete | 0.191 |
| ROCKET + LightGBM (2K) | Complete | 0.462 |
| ROCKET + LightGBM (5K) | Complete | 0.458 |

**Current best approach**: ROCKET + LightGBM with ~0.46 CV F1 score.

The feature-based approach substantially outperforms deep learning on this dataset, likely due to the limited training size and the effectiveness of random convolutional kernels for capturing time series patterns.

---

## References

- [MALLORN Dataset Paper](https://arxiv.org/abs/2512.04946)
- [ROCKET: Exceptionally fast and accurate time series classification](https://arxiv.org/abs/1910.13051)
- [TDE Review: Gezari 2021](https://arxiv.org/abs/2104.14580)
