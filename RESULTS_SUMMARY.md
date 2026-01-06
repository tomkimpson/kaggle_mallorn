# MALLORN TDE Classification - Results Summary

**Competition**: [Kaggle MALLORN Astronomical Classification Challenge](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)
**Task**: Binary classification of Tidal Disruption Events (TDEs)
**Metric**: F1 Score
**Last Updated**: 2026-01-07

---

## Leaderboard Results

| Model | CV F1 | Public LB | Notes |
|-------|-------|-----------|-------|
| **GP/DRW Features** | 0.556 | **0.5668** | New best! |
| ROCKET + Domain + Optuna | 0.544 | 0.5388 | Hyperparameter tuned |
| ROCKET + LightGBM (5K) | 0.458 | 0.4753 | Baseline ROCKET |
| ROCKET Ensemble | 0.387 | 0.4306 | Ridge classifier |
| Transformer | 0.191 | 0.1010 | Deep learning |
| Transformer Ensemble | 0.191 | 0.0850 | 5-fold ensemble |
| CNN | 0.166 | 0.0917 | Deep learning |

---

## Model Details

### 1. GP/DRW Features (Best Model)

**CV F1: 0.556 | Public LB: 0.5668**

Combines ROCKET features with physics-informed GP/DRW modeling:
- **60,000 ROCKET features** (5K kernels x 6 bands x 2)
- **138 domain features** (per-band statistics, colors)
- **32 GP/DRW features** (Damped Random Walk + Gaussian Process fits)

Key insight: GP/DRW features exploit how MALLORN was generated - TDEs use GP with Matern kernel, AGN use DRW. The likelihood ratio is highly discriminative.

Per-fold breakdown:
| Fold | F1 | ROC-AUC |
|------|-----|---------|
| 0 | 0.522 | 0.906 |
| 1 | 0.538 | 0.931 |
| 2 | 0.640 | 0.972 |
| 3 | 0.542 | 0.955 |
| 4 | 0.537 | 0.935 |

### 2. ROCKET + Domain + Optuna

**CV F1: 0.544 | Public LB: 0.5388**

Hyperparameter-tuned LightGBM on ROCKET + domain features:
- 20 Optuna trials
- Best params: num_leaves=40, lr=0.028, feature_fraction=0.35

Per-fold breakdown:
| Fold | F1 | ROC-AUC |
|------|-----|---------|
| 0 | 0.423 | 0.902 |
| 1 | 0.516 | 0.912 |
| 2 | 0.621 | 0.947 |
| 3 | 0.600 | 0.947 |
| 4 | 0.559 | 0.946 |

### 3. ROCKET + Domain (Baseline)

**CV F1: 0.481 | Public LB: N/A**

Added 138 physics-informed features to ROCKET:
- Power-law decay fitting (TDEs decay as t^-5/3)
- Color evolution (g-r, u-g colors over time)
- Smoothness metrics (TDEs are smooth, SNe have bumps)
- Signal quality (SNR, coverage)

---

## Worktree Experiments

### Bagged CV (exp/bagging-repeated-cv)

**Pooled F1: 0.488 | Mean F1: 0.503 ± 0.059**

- 3 repeats x 5 folds x 3 bags = 45 models
- Reduced variance but no improvement in peak performance
- Threshold optimized on pooled OOF predictions

### Bin-Dependent Threshold (exp/bin-threshold)

**CV F1: 0.481**

- Tried per-bin thresholds by n_obs, SNR, redshift
- Only marginal improvement (+0.003)
- Not worth the complexity

### MultiROCKET (exp/multirocket)

**Status: Timed out - needs re-run**

- Custom Python implementation was too slow
- Fix: Install sktime properly for vectorized kernels

---

## Key Findings

1. **Feature-based methods dominate**: ROCKET + LightGBM achieves 3-5x better F1 than deep learning
2. **Domain features help**: Adding physics-informed features improved F1 by ~5%
3. **GP/DRW features are powerful**: Exploiting the data generation process gives another ~10% boost
4. **CV correlates with LB**: Models generalize well (CV ~= LB scores)
5. **High fold variance**: F1 ranges 0.42-0.64 across folds due to limited TDE samples

---

## Dataset Statistics

| Split | Objects | TDEs | Non-TDEs | TDE Fraction |
|-------|---------|------|----------|--------------|
| Train | 3,043 | 148 | 2,895 | 4.9% |
| Test | 7,135 | ? | ? | ? |

---

## Infrastructure

- **Compute**: OzSTAR SLURM cluster
- **Framework**: LightGBM, scikit-learn, PyTorch
- **Training time**: ~1-2 hours for full pipeline
- **Task tracking**: `bd` (beads)

---

## Files

### Submissions
- `submission_gp_drw.csv` - GP/DRW features (LB 0.5668)
- `submission_rocket_domain_optuna.csv` - Optuna tuned (LB 0.5388)
- `submission_rocket_lgbm_5k.csv` - ROCKET baseline (LB 0.4753)

### Checkpoints
- `checkpoints/rocket_domain_optuna/` - Optuna results
- `checkpoints/rocket_domain_5k/` - Domain features
- `checkpoints/gp_drw/` - GP/DRW model (in worktree)

### Scripts
- `scripts/train_rocket_lgbm_gp_drw.py` - GP/DRW training
- `scripts/train_rocket_lgbm.py` - Main ROCKET pipeline
- `src/features/drw_features.py` - DRW/GP feature extraction

---

## Next Steps

1. **Ensemble GP/DRW + Optuna models** - Combine best approaches
2. **Complete MultiROCKET experiment** - Different feature space for diversity
3. **Semi-supervised learning** - Use test predictions for pseudo-labeling
4. **Repeated CV + bagging on GP/DRW** - Reduce variance on best model
