# MALLORN: Tidal Disruption Event Classification

Deep learning approach for identifying Tidal Disruption Events (TDEs) from LSST-simulated light curves.

**Competition**: [Kaggle MALLORN Astronomical Classification Challenge](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)

## Problem Overview

Tidal Disruption Events occur when a star is torn apart by a supermassive black hole. With only ~100 observed TDEs to date, they remain scientifically valuable but rare. The upcoming LSST survey will discover many more transients, but we need ML methods to identify TDEs from photometric data alone.

**Task**: Binary classification - TDE (1) vs non-TDE (0)
**Metric**: F1 score
**Challenge**: Severe class imbalance (~5% TDEs in training data)

## Dataset

| Split | Objects | TDEs | Observations |
|-------|---------|------|--------------|
| Train | 3,043 | 148 (4.9%) | 479,384 |
| Test | 7,135 | ? | 1,145,125 |

Each object has:
- **Light curves**: Time-series flux measurements in 6 LSST bands (u, g, r, i, z, y)
- **Metadata**: Redshift (Z), extinction coefficient (EBV)

Other classes include: AGN (dominant), SN Ia, SN II, SN Ib/c, SLSN-I/II, and more.

## Approach

### Architecture

We use per-band encoders that process each photometric filter's time series independently, then fuse the representations for classification.

```
Input: 6 bands × variable-length sequences × 3 features (flux, flux_err, time_delta)
                              ↓
┌─────────────────────────────────────────────────────────┐
│  Per-Band Encoders (CNN or Transformer)                 │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │  u  │ │  g  │ │  r  │ │  i  │ │  z  │ │  y  │       │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘       │
│     └───────┴───────┴───┬───┴───────┴───────┘           │
│                         ↓                               │
│              Concatenate + Metadata (Z, EBV)            │
│                         ↓                               │
│                  Classification MLP                     │
│                         ↓                               │
│                   P(TDE) → Binary                       │
└─────────────────────────────────────────────────────────┘
```

### Models

| Model | Parameters | Description |
|-------|------------|-------------|
| `LightCurveCNN` | ~545K | 1D CNN with 4 conv layers per band, global pooling |
| `LightCurveTransformer` | ~175K | Transformer encoder per band with cross-band attention |

### Handling Class Imbalance

1. **Focal Loss**: Down-weights easy negatives, focuses on hard examples
   - α = 0.75 (weight for positive class)
   - γ = 2.0 (focusing parameter)

2. **Weighted Sampling**: Oversample TDEs during training

3. **Threshold Optimization**: Find optimal decision threshold on validation set

### Data Preprocessing

1. **De-extinction**: Correct for Milky Way dust using Fitzpatrick99 law
2. **NaN Handling**: Remove observations with missing flux values
3. **Normalization**: Per-object flux normalization (divide by max absolute value)
4. **Padding**: Fixed sequence length (200) with masking for variable-length inputs

## Project Structure

```
kaggle_mallorn/
├── data/                          # Competition data
│   ├── train_log.csv             # Training metadata + labels
│   ├── test_log.csv              # Test metadata
│   ├── split_01-20/              # Light curve files
│   └── sample_submission.csv
├── src/
│   ├── data/
│   │   ├── loader.py             # Data loading utilities
│   │   ├── preprocessing.py      # De-extinction, normalization
│   │   └── dataset.py            # PyTorch Dataset
│   ├── models/
│   │   ├── cnn.py                # 1D CNN architecture
│   │   └── transformer.py        # Transformer architecture
│   └── training/
│       ├── losses.py             # Focal loss, weighted BCE
│       ├── metrics.py            # F1, threshold optimization
│       └── trainer.py            # Training loop
├── scripts/
│   ├── train.py                  # Training entry point
│   └── submit.py                 # Generate submission
├── slurm/                        # SLURM job scripts
├── checkpoints/                  # Saved models
└── requirements.txt
```

## Usage

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd kaggle_mallorn

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download competition data (requires Kaggle API)
# First: pip install kaggle && configure ~/.kaggle/kaggle.json
kaggle competitions download -c mallorn-astronomical-classification-challenge
unzip mallorn-astronomical-classification-challenge.zip -d data/
rm mallorn-astronomical-classification-challenge.zip
```

### Training

```bash
# Quick test (single fold, 10 epochs)
python scripts/train.py --model cnn --epochs 10 --fold 0

# Full 5-fold cross-validation
python scripts/train.py --model cnn --epochs 100 --batch_size 64

# Transformer model
python scripts/train.py --model transformer --epochs 100 --lr 5e-4
```

**Key Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | cnn | Model architecture (cnn, cnn_attn, transformer) |
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--loss` | focal | Loss function (focal, bce) |
| `--fold` | None | Specific fold (None = all folds) |
| `--device` | auto | Device (auto, cpu, cuda, mps) |

### Generate Submission

```bash
# Single model
python scripts/submit.py --checkpoint checkpoints/cnn/fold_0/best_model.pt

# Ensemble all folds
python scripts/submit.py --checkpoint_dir checkpoints/cnn --ensemble
```

### SLURM (GPU Cluster)

```bash
# Update path in script first
sbatch slurm/train_single.sh   # Quick test
sbatch slurm/train_mallorn.sh  # Full training
```

## Results

*To be updated after full training*

| Model | CV F1 | Precision | Recall | Threshold |
|-------|-------|-----------|--------|-----------|
| CNN | - | - | - | - |
| Transformer | - | - | - | - |
| Ensemble | - | - | - | - |

## Future Improvements

- [ ] Data augmentation (noise injection, time warping)
- [ ] Hand-crafted features (power-law decay fitting, color evolution)
- [ ] Multi-task learning (binary + multi-class)
- [ ] Pseudo-labeling on high-confidence test predictions
- [ ] Ensemble with gradient boosting models

## References

- [MALLORN Dataset Paper](https://arxiv.org/abs/2512.04946)
- [TDE Review: Gezari 2021](https://arxiv.org/abs/2104.14580)
- [TDE Review: van Velzen et al. 2020](https://arxiv.org/abs/2008.05461)
