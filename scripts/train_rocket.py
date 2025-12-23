#!/usr/bin/env python3
"""
Train ROCKET classifier on GP-interpolated light curves.

Pipeline:
1. GP interpolate each light curve to regular time grid
2. Apply ROCKET transform to extract features
3. Train Ridge classifier (or LogisticRegression) with CV
4. Generate predictions

Usage:
    python scripts/train_rocket.py --num_kernels 1000 --grid_points 100
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import joblib
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_train_data, load_test_data
from src.features.rocket import MultiChannelROCKET
from src.features.gp_interpolation import GPInterpolator

BANDS = ['u', 'g', 'r', 'i', 'z', 'y']


def parse_args():
    parser = argparse.ArgumentParser(description='Train ROCKET on GP-interpolated light curves')
    parser.add_argument('--num_kernels', type=int, default=1000,
                        help='Number of ROCKET kernels per band')
    parser.add_argument('--grid_points', type=int, default=100,
                        help='Number of time points in interpolated grid')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--classifier', type=str, default='ridge',
                        choices=['ridge', 'logistic'],
                        help='Classifier type')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/rocket',
                        help='Directory to save models')
    parser.add_argument('--output', type=str, default='submission_rocket.csv',
                        help='Output submission file')
    parser.add_argument('--skip_gp', action='store_true',
                        help='Skip GP, use simple linear interpolation (faster)')
    return parser.parse_args()


def interpolate_lightcurve_simple(
    obj_lc: pd.DataFrame,
    n_points: int = 100,
) -> np.ndarray:
    """
    Simple linear interpolation to regular grid (faster than GP).

    Args:
        obj_lc: Light curve DataFrame for single object
        n_points: Number of output time points

    Returns:
        Array of shape (6, n_points) - interpolated flux per band
    """
    result = np.zeros((6, n_points))

    # Get time range across all bands
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

        # Remove NaN
        valid = ~(np.isnan(t) | np.isnan(f))
        t, f = t[valid], f[valid]

        if len(t) < 2:
            continue

        # Normalize flux
        f_scale = np.max(np.abs(f)) if np.max(np.abs(f)) > 0 else 1.0
        f = f / f_scale

        # Linear interpolation
        result[i] = np.interp(t_grid, t, f, left=0, right=0)

    return result


def interpolate_lightcurve_gp(
    obj_lc: pd.DataFrame,
    interpolator: GPInterpolator,
    n_points: int = 100,
) -> np.ndarray:
    """
    GP interpolation to regular grid.

    Args:
        obj_lc: Light curve DataFrame for single object
        interpolator: GP interpolator instance
        n_points: Number of output time points

    Returns:
        Array of shape (6, n_points) - interpolated flux per band
    """
    result = np.zeros((6, n_points))

    # Get time range
    all_times = obj_lc['Time (MJD)'].dropna()
    if len(all_times) == 0:
        return result

    t_min, t_max = all_times.min(), all_times.max()
    if t_max <= t_min:
        return result

    t_grid = np.linspace(t_min, t_max, n_points)

    for i, band in enumerate(BANDS):
        band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')
        if len(band_data) < 3:  # Need at least 3 points for GP
            # Fall back to linear
            if len(band_data) >= 2:
                t = band_data['Time (MJD)'].values
                f = band_data['Flux'].values
                valid = ~(np.isnan(t) | np.isnan(f))
                t, f = t[valid], f[valid]
                if len(t) >= 2:
                    f_scale = np.max(np.abs(f)) if np.max(np.abs(f)) > 0 else 1.0
                    result[i] = np.interp(t_grid, t, f / f_scale, left=0, right=0)
            continue

        t = band_data['Time (MJD)'].values
        f = band_data['Flux'].values
        f_err = band_data['Flux_err'].values

        # Remove NaN
        valid = ~(np.isnan(t) | np.isnan(f) | np.isnan(f_err))
        t, f, f_err = t[valid], f[valid], f_err[valid]

        if len(t) < 3:
            continue

        # Normalize
        f_scale = np.max(np.abs(f)) if np.max(np.abs(f)) > 0 else 1.0
        f = f / f_scale
        f_err = f_err / f_scale

        try:
            # Fit GP
            gp_result = interpolator._fit_single_band(t, f, f_err, t_grid)
            if gp_result.success:
                result[i] = gp_result.flux_mean
            else:
                # Fall back to linear
                result[i] = np.interp(t_grid, t, f, left=0, right=0)
        except Exception:
            result[i] = np.interp(t_grid, t, f, left=0, right=0)

    return result


def prepare_data(
    log_df: pd.DataFrame,
    lc_df: pd.DataFrame,
    n_points: int = 100,
    use_gp: bool = False,
    desc: str = "Interpolating",
) -> np.ndarray:
    """
    Prepare interpolated light curve data.

    Returns:
        Array of shape (n_samples, 6, n_points)
    """
    n_samples = len(log_df)
    X = np.zeros((n_samples, 6, n_points))

    if use_gp:
        interpolator = GPInterpolator(n_restarts=1)
    else:
        interpolator = None

    for idx, row in tqdm(log_df.iterrows(), total=n_samples, desc=desc):
        obj_id = row['object_id']
        obj_lc = lc_df[lc_df['object_id'] == obj_id]

        if use_gp:
            X[idx] = interpolate_lightcurve_gp(obj_lc, interpolator, n_points)
        else:
            X[idx] = interpolate_lightcurve_simple(obj_lc, n_points)

    return X


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

    print("=" * 60)
    print("ROCKET Training for TDE Classification")
    print("=" * 60)
    print(f"Kernels per band: {args.num_kernels}")
    print(f"Grid points: {args.grid_points}")
    print(f"Classifier: {args.classifier}")
    print(f"Use GP: {not args.skip_gp}")

    # Load data
    print("\nLoading training data...")
    train_log, train_lc = load_train_data()
    print(f"Loaded {len(train_log)} training objects")

    # Prepare interpolated data
    print("\nInterpolating light curves...")
    X = prepare_data(
        train_log, train_lc,
        n_points=args.grid_points,
        use_gp=not args.skip_gp,
        desc="Interpolating train"
    )
    y = train_log['target'].values

    print(f"Interpolated data shape: {X.shape}")
    print(f"Class distribution: TDE={y.sum()}, non-TDE={len(y) - y.sum()}")

    # Cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_results = []
    oof_preds = np.zeros(len(y))
    rockets = []
    scalers = []
    classifiers = []

    print(f"\n{'=' * 60}")
    print(f"Starting {args.n_folds}-fold Cross-Validation")
    print(f"{'=' * 60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit ROCKET on training data
        print(f"Fitting ROCKET ({args.num_kernels} kernels per band)...")
        rocket = MultiChannelROCKET(
            num_kernels_per_channel=args.num_kernels,
            seed=args.seed + fold,
        )
        X_train_features = rocket.fit_transform(X_train)
        X_val_features = rocket.transform(X_val)

        print(f"ROCKET features shape: {X_train_features.shape}")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
        X_val_scaled = scaler.transform(X_val_features)

        # Train classifier
        print(f"Training {args.classifier} classifier...")
        if args.classifier == 'ridge':
            clf = RidgeClassifierCV(
                alphas=np.logspace(-3, 3, 10),
                cv=3,
            )
            clf.fit(X_train_scaled, y_train)
            # Ridge doesn't have predict_proba, use decision function
            val_decision = clf.decision_function(X_val_scaled)
            # Convert to probability-like scores
            val_preds_proba = 1 / (1 + np.exp(-val_decision))
        else:
            clf = LogisticRegressionCV(
                Cs=10,
                cv=3,
                class_weight='balanced',
                max_iter=1000,
                random_state=args.seed,
            )
            clf.fit(X_train_scaled, y_train)
            val_preds_proba = clf.predict_proba(X_val_scaled)[:, 1]

        oof_preds[val_idx] = val_preds_proba

        # Find optimal threshold
        threshold, best_f1 = find_optimal_threshold(y_val, val_preds_proba)
        val_preds = (val_preds_proba >= threshold).astype(int)

        # Metrics
        precision = precision_score(y_val, val_preds)
        recall = recall_score(y_val, val_preds)
        try:
            roc_auc = roc_auc_score(y_val, val_preds_proba)
        except:
            roc_auc = 0.5

        print(f"Fold {fold} Results:")
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

        rockets.append(rocket)
        scalers.append(scaler)
        classifiers.append(clf)

        # Save fold model
        fold_dir = checkpoint_dir / f'fold_{fold}'
        fold_dir.mkdir(exist_ok=True)
        joblib.dump({
            'rocket': rocket,
            'scaler': scaler,
            'classifier': clf,
            'threshold': threshold,
        }, fold_dir / 'model.pkl')

    # Summary
    print(f"\n{'=' * 60}")
    print("Cross-Validation Summary")
    print(f"{'=' * 60}")

    f1_scores = [r['f1'] for r in fold_results]
    avg_threshold = np.mean([r['threshold'] for r in fold_results])

    print(f"F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Precision: {np.mean([r['precision'] for r in fold_results]):.4f}")
    print(f"Recall: {np.mean([r['recall'] for r in fold_results]):.4f}")
    print(f"ROC-AUC: {np.mean([r['roc_auc'] for r in fold_results]):.4f}")
    print(f"Avg Threshold: {avg_threshold:.3f}")

    # Save summary
    summary = {
        'args': vars(args),
        'results': fold_results,
        'f1_mean': float(np.mean(f1_scores)),
        'f1_std': float(np.std(f1_scores)),
        'avg_threshold': float(avg_threshold),
    }
    joblib.dump(summary, checkpoint_dir / 'cv_summary.pkl')

    # Generate test predictions
    print(f"\n{'=' * 60}")
    print("Generating Test Predictions")
    print(f"{'=' * 60}")

    print("\nLoading test data...")
    test_log, test_lc = load_test_data()
    print(f"Loaded {len(test_log)} test objects")

    print("\nInterpolating test light curves...")
    X_test = prepare_data(
        test_log, test_lc,
        n_points=args.grid_points,
        use_gp=not args.skip_gp,
        desc="Interpolating test"
    )

    # Ensemble predictions
    test_preds_proba = np.zeros(len(X_test))

    for fold, (rocket, scaler, clf) in enumerate(zip(rockets, scalers, classifiers)):
        print(f"Predicting with fold {fold} model...")
        X_test_features = rocket.transform(X_test)
        X_test_scaled = scaler.transform(X_test_features)

        if args.classifier == 'ridge':
            decision = clf.decision_function(X_test_scaled)
            proba = 1 / (1 + np.exp(-decision))
        else:
            proba = clf.predict_proba(X_test_scaled)[:, 1]

        test_preds_proba += proba

    test_preds_proba /= args.n_folds
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

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
