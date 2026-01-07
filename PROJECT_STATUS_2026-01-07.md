# MALLORN TDE Classification - Project Status

**Date:** 2026-01-07
**Current Best Score:** 0.6117 (Public LB)
**Competition:** [MALLORN Astronomical Classification Challenge](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)

---

## Project Overview

The goal is to classify astronomical transients as **TDE (Tidal Disruption Events)** vs **non-TDE (mostly AGN)** using multi-band light curve data from the MALLORN simulated dataset.

- **Training data:** 3,043 objects (148 TDE, 2,895 non-TDE) - highly imbalanced (~4.9% positive)
- **Test data:** 7,135 objects
- **Bands:** 6 LSST filters (u, g, r, i, z, y)
- **Metric:** F1 score

---

## Key Discovery: Simpler is Better

The MALLORN dataset was generated using specific models:
- **TDEs:** Generated with GP (Gaussian Process) Matern kernel
- **AGN:** Generated with DRW (Damped Random Walk) model

By exploiting this "generator-aware" knowledge and fitting both GP and DRW models to each light curve, we can compute likelihood ratios that directly discriminate between classes.

**Critical finding:** The simpler model using only domain + GP/DRW features (237 features) **significantly outperforms** models with 60K+ ROCKET features.

---

## Model Evolution & Results

| Model | Features | CV F1 | Public LB | Notes |
|-------|----------|-------|-----------|-------|
| CNN baseline | Raw LC | 0.19 | 0.0917 | Neural net, poor |
| Transformer | Raw LC | 0.19 | 0.0850 | Neural net, poor |
| ROCKET only | 60K | 0.39 | 0.4306 | Time-series kernels |
| ROCKET + LightGBM | 60K | 0.46 | 0.4753 | Better classifier |
| ROCKET + Domain | 60K + 138 | 0.48 | - | Added hand-crafted features |
| ROCKET + Domain + Optuna | 60K + 138 | 0.544 | 0.5388 | Hyperparameter tuning |
| GP/DRW v1 | 60K + 138 + 30 | 0.556 | 0.5668 | First GP/DRW features |
| GP/DRW v2 | 60K + 168 + 69 | 0.548 | 0.5775 | Enhanced GP/DRW features |
| **No-ROCKET** | **168 + 69 = 237** | **0.608** | **0.6117** | **BEST - No ROCKET!** |

---

## Feature Engineering

### 1. Domain Features (~168 features)
Hand-crafted astronomical features per band:
- **Basic stats:** peak_flux, peak_time, mean, std, median, skewness, kurtosis
- **TDE-specific:** rise_time, decay_rate, amplitude_ratio, duration
- **Power-law decay:** Fits F = A × t^α post-peak (TDEs have α ≈ -5/3)
- **Smoothness:** inflection points, acceleration, monotonic ratio, roughness
- **Quality:** SNR metrics, n_obs, median_dt, max_gap
- **Colors:** g-r, r-i, i-z, u-g color indices and evolution

### 2. GP/DRW Features (~69 features)
Generator-aware features exploiting how MALLORN was created:

**Per-band (g, r, i):**
- DRW parameters: tau, sigma, SF_inf, log_likelihood
- GP Matern parameters: length_scale, amplitude, log_likelihood
- Likelihood ratio: DRW vs GP (positive = more AGN-like)
- Normalized versions: per-point likelihoods, AIC penalties

**Cross-band consistency (AGN are achromatic):**
- tau_cv_across_bands, sigma_cv_across_bands
- lr_mean, lr_std, lr_max, lr_sign_agreement
- combined_prefers_drw

**Rest-frame corrections:**
- tau_rest, gp_length_scale_rest (divided by 1+z)

### 3. ROCKET Features (~60,000 features)
Random Convolutional Kernel Transform - automated time-series feature extraction.
- **Finding:** These appear to add noise rather than signal for this problem.

---

## Current Codebase Structure

```
kaggle_mallorn/
├── src/
│   ├── data/loader.py          # Load train/test data from splits
│   ├── features/
│   │   ├── rocket.py           # ROCKET implementation
│   │   ├── light_curve_features.py  # Domain features
│   │   └── drw_features.py     # GP/DRW features
│   └── models/                 # CNN, Transformer (not used)
├── scripts/
│   ├── train_rocket_lgbm.py    # ROCKET + LightGBM (with Optuna)
│   ├── train_rocket_lgbm_gp_drw.py  # Full pipeline (with Optuna)
│   ├── train_domain_gpdrw_only.py   # No-ROCKET model (BEST)
│   ├── blend_predictions.py    # Weight sweep blending
│   └── oof_stacker.py          # Logistic regression meta-learner
├── slurm/                      # SLURM job scripts
├── checkpoints/                # Saved models and results
│   ├── gp_drw_v2/             # Previous best
│   ├── no_rocket/             # Current best
│   └── stacker/               # Ensemble meta-learner
└── blends/                     # Blended submission files
```

---

## Key Files for the Best Model

**Training script:** `scripts/train_domain_gpdrw_only.py`
```bash
python scripts/train_domain_gpdrw_only.py \
    --use_domain_features \
    --use_gp_drw_features \
    --gp_drw_bands g,r,i \
    --checkpoint_dir checkpoints/no_rocket \
    --output submission_no_rocket.csv
```

**Outputs:**
- `checkpoints/no_rocket/cv_summary.yaml` - CV results
- `checkpoints/no_rocket/oof_predictions.npz` - OOF predictions for stacking
- `checkpoints/no_rocket/test_predictions.npz` - Test predictions
- `submission_no_rocket.csv` - Binary predictions
- `submission_no_rocket_probs.csv` - Probability predictions

---

## Experiments Tried

### Blending (v1 + v2)
Tested weight sweep from 0.0 to 1.0 between rocket_domain_optuna (v1) and gp_drw_v2 (v2).
- **Result:** Blending hurt performance. Pure v2 (0.5775) beat all blends.
- **Lesson:** Models weren't complementary enough.

### Optuna Hyperparameter Tuning
Added to `train_rocket_lgbm_gp_drw.py`:
- F1 objective with pooled OOF threshold selection
- Extended search space (DART boosting, higher regularization)
- **Status:** Job timed out before completing trials (feature extraction too slow)

### OOF Stacking
Combined gp_drw_v2 + no_rocket with LogisticRegressionCV:
- Meta-learner OOF F1: 0.592
- Weights: no_rocket (0.19) > gp_drw_v2 (0.13)
- **Status:** Kaggle submission hit rate limit, not yet tested on LB

### MultiROCKET (Still Running)
Job 8177804 testing MultiROCKET variant with multivariate kernels.
- Running for 11+ hours
- May not be valuable given no-ROCKET model's success

---

## What Works

1. **GP/DRW likelihood ratios** - Directly exploit the data generation process
2. **Domain features** - Physics-informed (TDE decay, colors, smoothness)
3. **Simpler models** - 237 features beats 60K features
4. **LightGBM with class weighting** - Handles imbalance well
5. **Threshold optimization** - F1-optimal threshold on OOF predictions

## What Doesn't Work

1. **ROCKET features** - Add noise, hurt generalization
2. **Neural networks** - CNN/Transformer failed badly
3. **Simple blending** - Didn't improve over best single model
4. **High complexity** - More features ≠ better performance

---

## Next Steps (Recommended)

### High Priority
1. **Optuna on no-ROCKET model** - Tune hyperparameters for the best model
2. **Re-run stacker submission** - Rate limit should reset
3. **Feature ablation** - Which GP/DRW features matter most?

### Medium Priority
4. **Repeated stratified CV** - 5-fold × 3 repeats for stable threshold
5. **Add u-band GP/DRW** - Selectively when n_obs_u >= 15
6. **Stratify by data quality** - Split by n_obs or time_span buckets

### Lower Priority
7. **Cancel/monitor multirocket** - Likely not needed now
8. **Try XGBoost/CatBoost** - Alternative gradient boosting

---

## How to Submit to Kaggle

```bash
source venv/bin/activate
export KAGGLE_API_TOKEN=KGAT_209f2e6f5fd3bc6c81588500c431b21c
kaggle competitions submit -c mallorn-astronomical-classification-challenge \
    -f submission_no_rocket.csv -m "Description"
```

---

## Running Jobs

As of 2026-01-07 21:00 AEDT:
- **8177804** (multirocket) - Still running on john18 (~12 hours)

Completed:
- **8183409** (optuna_gp_drw) - TIMED OUT after 6 hours
- **8183410** (no_rocket) - COMPLETED, produced best result

---

## Contact / Resources

- **Codebase:** `/fred/oz022/tkimpson/kaggle_mallorn/`
- **MALLORN Paper:** arXiv:2512.04946
- **Kaggle Competition:** mallorn-astronomical-classification-challenge
