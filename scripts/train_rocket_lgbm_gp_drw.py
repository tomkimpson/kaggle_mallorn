#!/usr/bin/env python3
"""
ROCKET + GP/DRW Features + LightGBM for MALLORN TDE Classification.

Adds generator-aware features exploiting how MALLORN was created:
- GP Matern kernel for transient modeling
- Damped Random Walk (DRW) for AGN variability
- Likelihood ratio to discriminate AGN from TDEs

Reference: MALLORN paper (arXiv:2512.04946)

Usage:
    python scripts/train_rocket_lgbm_gp_drw.py \
        --load_features /path/to/train_features.npz \
        --use_gp_drw_features
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import lightgbm as lgb
import joblib
from tqdm import tqdm
import sys
import yaml
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_train_data, load_test_data
from src.features.rocket import MultiChannelROCKET
from src.features.light_curve_features import extract_features_batch, get_feature_columns
from src.features.drw_features import (
    extract_gp_drw_features_batch,
    get_gp_drw_feature_columns
)

BANDS = ['u', 'g', 'r', 'i', 'z', 'y']


def parse_args():
    parser = argparse.ArgumentParser(description='ROCKET + GP/DRW + LightGBM')

    # ROCKET parameters
    parser.add_argument('--num_kernels', type=int, default=5000,
                        help='Number of ROCKET kernels per band')
    parser.add_argument('--grid_points', type=int, default=100,
                        help='Number of time points in interpolated grid')

    # Cross-validation
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # LightGBM parameters
    parser.add_argument('--num_boost_round', type=int, default=1000,
                        help='Maximum boosting rounds')
    parser.add_argument('--early_stopping_rounds', type=int, default=50,
                        help='Early stopping patience')

    # Feature options
    parser.add_argument('--use_domain_features', action='store_true',
                        help='Include domain-specific features')
    parser.add_argument('--use_gp_drw_features', action='store_true',
                        help='Include GP/DRW model features')
    parser.add_argument('--gp_drw_bands', type=str, default='g,r,i',
                        help='Bands to use for GP/DRW fitting (comma-separated)')

    # Caching
    parser.add_argument('--load_features', type=str, default=None,
                        help='Load pre-computed ROCKET features')
    parser.add_argument('--load_gp_drw_features', type=str, default=None,
                        help='Load pre-computed GP/DRW features')
    parser.add_argument('--save_gp_drw_features', action='store_true',
                        help='Save GP/DRW features after extraction')

    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/gp_drw',
                        help='Directory to save models')
    parser.add_argument('--output', type=str, default='submission_gp_drw.csv',
                        help='Output submission file')

    return parser.parse_args()


def interpolate_lightcurve(obj_lc: pd.DataFrame, n_points: int = 100) -> np.ndarray:
    """Linear interpolation to regular grid."""
    result = np.zeros((6, n_points))

    all_times = obj_lc['Time (MJD)'].dropna()
    if len(all_times) == 0:
        return result

    t_min, t_max = all_times.min(), all_times.max()
    if t_max <= t_min:
        return result

    t_grid = np.linspace(t_min, t_max, n_points)

    for i, band in enumerate(BANDS):
        band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')
        if len(band_data) < 2:
            continue

        t = band_data['Time (MJD)'].values
        f = band_data['Flux'].values

        valid = ~(np.isnan(t) | np.isnan(f))
        t, f = t[valid], f[valid]

        if len(t) < 2:
            continue

        f_scale = np.max(np.abs(f)) if np.max(np.abs(f)) > 0 else 1.0
        f = f / f_scale

        result[i] = np.interp(t_grid, t, f, left=0, right=0)

    return result


def prepare_interpolated_data(log_df: pd.DataFrame, lc_df: pd.DataFrame,
                               n_points: int, desc: str = "Interpolating") -> np.ndarray:
    """Prepare interpolated light curve data."""
    n_samples = len(log_df)
    X = np.zeros((n_samples, 6, n_points))

    lc_groups = lc_df.groupby('object_id')

    for idx, row in tqdm(log_df.iterrows(), total=n_samples, desc=desc):
        obj_id = row['object_id']
        try:
            obj_lc = lc_groups.get_group(obj_id)
        except KeyError:
            continue
        X[idx] = interpolate_lightcurve(obj_lc, n_points)

    return X


def extract_domain_features(log_df: pd.DataFrame, lc_df: pd.DataFrame,
                           desc: str = "Extracting domain features") -> np.ndarray:
    """Extract domain-specific features."""
    print(f"\n{desc}...")
    features_df = extract_features_batch(log_df, lc_df, verbose=True)

    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in features_df.columns]

    print(f"Extracted {len(available_cols)} domain features")

    X_domain = features_df[available_cols].values.astype(np.float32)
    X_domain = np.nan_to_num(X_domain, nan=0.0, posinf=0.0, neginf=0.0)

    return X_domain, available_cols


def extract_gp_drw_features(log_df: pd.DataFrame, lc_df: pd.DataFrame,
                            bands: list, desc: str = "Extracting GP/DRW features") -> np.ndarray:
    """Extract GP and DRW model features."""
    print(f"\n{desc}...")
    print(f"Fitting GP (Matern) and DRW models on bands: {bands}")
    print("Note: This may take several minutes...")

    features_df = extract_gp_drw_features_batch(log_df, lc_df, bands=bands, verbose=True)

    feature_cols = get_gp_drw_feature_columns(bands)
    available_cols = [c for c in feature_cols if c in features_df.columns]

    print(f"Extracted {len(available_cols)} GP/DRW features")

    X_gp_drw = features_df[available_cols].values.astype(np.float32)
    X_gp_drw = np.nan_to_num(X_gp_drw, nan=0.0, posinf=0.0, neginf=0.0)

    return X_gp_drw, available_cols


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> tuple:
    """Find optimal threshold that maximizes F1 score."""
    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def main():
    args = parse_args()
    np.random.seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    gp_drw_bands = args.gp_drw_bands.split(',')

    print("=" * 70)
    print("ROCKET + GP/DRW Features + LightGBM Training")
    print("=" * 70)
    print(f"Kernels per band: {args.num_kernels}")
    print(f"Domain features: {args.use_domain_features}")
    print(f"GP/DRW features: {args.use_gp_drw_features}")
    print(f"GP/DRW bands: {gp_drw_bands}")

    # Load data
    print("\nLoading training data...")
    train_log, train_lc = load_train_data()
    y = train_log['target'].values
    print(f"Loaded {len(train_log)} training objects")
    print(f"Class distribution: TDE={y.sum()}, non-TDE={len(y) - y.sum()}")

    # Prepare ROCKET features
    if args.load_features:
        print(f"\nLoading pre-computed ROCKET features from {args.load_features}...")
        cached = np.load(args.load_features)
        X_rocket = cached['X_rocket']
        print(f"Loaded ROCKET features shape: {X_rocket.shape}")
    else:
        print("\nInterpolating training light curves...")
        X_interp = prepare_interpolated_data(
            train_log, train_lc,
            n_points=args.grid_points,
            desc="Interpolating train"
        )

        print(f"\nFitting ROCKET with {args.num_kernels} kernels per band...")
        rocket = MultiChannelROCKET(
            num_kernels_per_channel=args.num_kernels,
            seed=args.seed,
        )
        X_rocket = rocket.fit_transform(X_interp)
        print(f"ROCKET features shape: {X_rocket.shape}")

        joblib.dump(rocket, checkpoint_dir / 'rocket_transformer.pkl')

    # Combine all feature sets
    feature_sets = [X_rocket]
    all_feature_cols = []

    # Domain features
    domain_feature_cols = None
    if args.use_domain_features:
        X_domain, domain_feature_cols = extract_domain_features(
            train_log, train_lc, desc="Extracting training domain features"
        )
        print(f"Domain features shape: {X_domain.shape}")
        feature_sets.append(X_domain)
        all_feature_cols.extend(domain_feature_cols)

    # GP/DRW features
    gp_drw_feature_cols = None
    if args.use_gp_drw_features:
        if args.load_gp_drw_features:
            print(f"\nLoading GP/DRW features from {args.load_gp_drw_features}...")
            gp_drw_cached = np.load(args.load_gp_drw_features)
            X_gp_drw = gp_drw_cached['X_gp_drw']
            gp_drw_feature_cols = list(gp_drw_cached.get('feature_cols', []))
        else:
            X_gp_drw, gp_drw_feature_cols = extract_gp_drw_features(
                train_log, train_lc, bands=gp_drw_bands,
                desc="Extracting training GP/DRW features"
            )

            if args.save_gp_drw_features:
                gp_drw_path = checkpoint_dir / 'gp_drw_features.npz'
                np.savez_compressed(gp_drw_path, X_gp_drw=X_gp_drw,
                                   feature_cols=gp_drw_feature_cols)
                print(f"GP/DRW features saved to {gp_drw_path}")

        print(f"GP/DRW features shape: {X_gp_drw.shape}")
        feature_sets.append(X_gp_drw)
        all_feature_cols.extend(gp_drw_feature_cols)

    # Combine all features
    X_combined = np.hstack(feature_sets)
    print(f"\nCombined features shape: {X_combined.shape}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # LightGBM parameters
    pos_weight = (len(y) - y.sum()) / y.sum()
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'scale_pos_weight': pos_weight,
        'verbosity': -1,
        'seed': args.seed,
    }

    # Cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    oof_preds = np.zeros(len(y))
    fold_results = []
    models = []

    print(f"\n{'=' * 70}")
    print(f"Starting {args.n_folds}-fold Cross-Validation")
    print(f"{'=' * 70}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        print(f"\n--- Fold {fold} ---")

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_set = lgb.Dataset(X_train, y_train)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set)

        model = lgb.train(
            params, train_set,
            num_boost_round=args.num_boost_round,
            valid_sets=[train_set, val_set],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(args.early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ],
        )

        val_preds_proba = model.predict(X_val)
        oof_preds[val_idx] = val_preds_proba

        threshold, best_f1 = find_optimal_threshold(y_val, val_preds_proba)
        val_preds = (val_preds_proba >= threshold).astype(int)

        precision = precision_score(y_val, val_preds)
        recall = recall_score(y_val, val_preds)
        roc_auc = roc_auc_score(y_val, val_preds_proba)

        print(f"\nFold {fold} Results:")
        print(f"  F1: {best_f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Threshold: {threshold:.3f}")

        fold_results.append({
            'fold': fold,
            'f1': best_f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'threshold': threshold,
        })

        # Save fold model
        fold_dir = checkpoint_dir / f'fold_{fold}'
        fold_dir.mkdir(exist_ok=True)
        model.save_model(str(fold_dir / 'model.txt'))

        models.append(model)

    # Summary
    print(f"\n{'=' * 70}")
    print("Cross-Validation Summary")
    print(f"{'=' * 70}")

    f1_scores = [r['f1'] for r in fold_results]
    avg_threshold = np.mean([r['threshold'] for r in fold_results])

    summary = {
        'f1_mean': float(np.mean(f1_scores)),
        'f1_std': float(np.std(f1_scores)),
        'precision_mean': float(np.mean([r['precision'] for r in fold_results])),
        'recall_mean': float(np.mean([r['recall'] for r in fold_results])),
        'roc_auc_mean': float(np.mean([r['roc_auc'] for r in fold_results])),
        'avg_threshold': float(avg_threshold),
        'n_rocket_features': X_rocket.shape[1],
        'n_domain_features': len(domain_feature_cols) if domain_feature_cols else 0,
        'n_gp_drw_features': len(gp_drw_feature_cols) if gp_drw_feature_cols else 0,
        'n_total_features': X_combined.shape[1],
    }

    print(f"F1: {summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}")
    print(f"Precision: {summary['precision_mean']:.4f}")
    print(f"Recall: {summary['recall_mean']:.4f}")
    print(f"ROC-AUC: {summary['roc_auc_mean']:.4f}")
    print(f"Average Threshold: {summary['avg_threshold']:.3f}")
    print(f"\nFeature breakdown:")
    print(f"  ROCKET: {summary['n_rocket_features']}")
    print(f"  Domain: {summary['n_domain_features']}")
    print(f"  GP/DRW: {summary['n_gp_drw_features']}")
    print(f"  Total: {summary['n_total_features']}")

    # Save summary
    with open(checkpoint_dir / 'cv_summary.yaml', 'w') as f:
        yaml.dump({
            'args': vars(args),
            'params': params,
            'results': fold_results,
            'summary': summary,
            'domain_feature_cols': domain_feature_cols,
            'gp_drw_feature_cols': gp_drw_feature_cols,
        }, f)

    joblib.dump(scaler, checkpoint_dir / 'scaler.pkl')

    # Save OOF predictions for ensemble
    np.savez_compressed(
        checkpoint_dir / 'oof_predictions.npz',
        oof_preds=oof_preds,
        y_true=y,
        optimal_threshold=avg_threshold,
    )

    # Feature importance for GP/DRW features
    if args.use_gp_drw_features and gp_drw_feature_cols:
        print(f"\n{'=' * 70}")
        print("GP/DRW Feature Importances")
        print(f"{'=' * 70}")

        importance = np.zeros(X_scaled.shape[1])
        for model in models:
            importance += model.feature_importance(importance_type='gain')
        importance /= len(models)

        # Find GP/DRW feature importance
        n_rocket = X_rocket.shape[1]
        n_domain = len(domain_feature_cols) if domain_feature_cols else 0
        gp_drw_start = n_rocket + n_domain

        gp_drw_importance = importance[gp_drw_start:]
        sorted_idx = np.argsort(gp_drw_importance)[::-1]

        for rank, idx in enumerate(sorted_idx[:10], 1):
            if idx < len(gp_drw_feature_cols):
                print(f"  {rank:2d}. {gp_drw_feature_cols[idx]}: {gp_drw_importance[idx]:.2f}")

    # Generate test predictions
    print(f"\n{'=' * 70}")
    print("Generating Test Predictions")
    print(f"{'=' * 70}")

    print("\nLoading test data...")
    test_log, test_lc = load_test_data()
    print(f"Loaded {len(test_log)} test objects")

    print("\nInterpolating test light curves...")
    X_test_interp = prepare_interpolated_data(
        test_log, test_lc,
        n_points=args.grid_points,
        desc="Interpolating test"
    )

    # Load ROCKET transformer
    if args.load_features:
        rocket = joblib.load(Path(args.load_features).parent / 'rocket_transformer.pkl')

    print("Applying ROCKET transform...")
    X_test_rocket = rocket.transform(X_test_interp)

    # Combine test features
    test_feature_sets = [X_test_rocket]

    if args.use_domain_features:
        X_test_domain, _ = extract_domain_features(
            test_log, test_lc, desc="Extracting test domain features"
        )
        test_feature_sets.append(X_test_domain)

    if args.use_gp_drw_features:
        X_test_gp_drw, _ = extract_gp_drw_features(
            test_log, test_lc, bands=gp_drw_bands,
            desc="Extracting test GP/DRW features"
        )
        test_feature_sets.append(X_test_gp_drw)

    X_test_combined = np.hstack(test_feature_sets)
    X_test_scaled = scaler.transform(X_test_combined)

    # Ensemble predictions
    test_preds_proba = np.zeros(len(X_test_scaled))
    for model in models:
        test_preds_proba += model.predict(X_test_scaled)
    test_preds_proba /= len(models)

    test_preds = (test_preds_proba >= avg_threshold).astype(int)

    print(f"\nUsing threshold: {avg_threshold:.3f}")
    print(f"Predicted TDEs: {test_preds.sum()}")
    print(f"TDE rate: {test_preds.mean():.2%}")

    # Create submission
    submission = pd.DataFrame({
        'object_id': test_log['object_id'],
        'prediction': test_preds,
    })

    submission.to_csv(args.output, index=False)
    print(f"\nSubmission saved to: {args.output}")

    # Save test predictions for ensemble
    np.savez_compressed(
        checkpoint_dir / 'test_predictions.npz',
        test_preds_proba=test_preds_proba,
        test_preds=test_preds,
        object_ids=test_log['object_id'].values,
    )

    print(f"\n{'=' * 70}")
    print("Training Complete!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
