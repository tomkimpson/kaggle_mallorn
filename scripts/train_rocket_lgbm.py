#!/usr/bin/env python3
"""
ROCKET + LightGBM Training Pipeline for MALLORN TDE Classification.

Improvements over baseline ROCKET:
1. LightGBM instead of Ridge - handles feature interactions + class imbalance
2. Support for more kernels (2000, 5000, 10000)
3. Feature caching for faster experimentation
4. Optuna hyperparameter optimization (optional)

Usage:
    python scripts/train_rocket_lgbm.py --num_kernels 5000
    python scripts/train_rocket_lgbm.py --num_kernels 5000 --optuna --optuna_trials 50
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

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_train_data, load_test_data
from src.features.rocket import MultiChannelROCKET
from src.features.light_curve_features import extract_features_batch, get_feature_columns

BANDS = ['u', 'g', 'r', 'i', 'z', 'y']


def parse_args():
    parser = argparse.ArgumentParser(description='ROCKET + LightGBM for TDE classification')

    # ROCKET parameters
    parser.add_argument('--num_kernels', type=int, default=5000,
                        help='Number of ROCKET kernels per band (default: 5000)')
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
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='Learning rate')
    parser.add_argument('--num_leaves', type=int, default=31,
                        help='Number of leaves')

    # Optuna
    parser.add_argument('--optuna', action='store_true',
                        help='Run Optuna hyperparameter optimization')
    parser.add_argument('--optuna_trials', type=int, default=50,
                        help='Number of Optuna trials')

    # Domain features
    parser.add_argument('--use_domain_features', action='store_true',
                        help='Combine ROCKET with domain-specific features (power-law decay, smoothness, etc.)')

    # Feature selection
    parser.add_argument('--feature_selection', action='store_true',
                        help='Use feature selection to reduce to top N features')
    parser.add_argument('--top_features', type=int, default=5000,
                        help='Number of top features to keep (default: 5000)')

    # Caching
    parser.add_argument('--save_features', action='store_true',
                        help='Save ROCKET features for reuse')
    parser.add_argument('--load_features', type=str, default=None,
                        help='Load pre-computed ROCKET features from path')

    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/rocket_lgbm',
                        help='Directory to save models')
    parser.add_argument('--output', type=str, default='submission_rocket_lgbm.csv',
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

        # Normalize flux
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
    """Extract domain-specific features (power-law decay, smoothness, colors, etc.)."""
    print(f"\n{desc}...")
    features_df = extract_features_batch(log_df, lc_df, verbose=True)

    # Get feature columns (excluding object_id and target)
    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in features_df.columns]

    print(f"Extracted {len(available_cols)} domain features")

    X_domain = features_df[available_cols].values.astype(np.float32)
    X_domain = np.nan_to_num(X_domain, nan=0.0, posinf=0.0, neginf=0.0)

    return X_domain, available_cols


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


def run_optuna_optimization(X: np.ndarray, y: np.ndarray,
                            n_folds: int, n_trials: int, seed: int) -> dict:
    """Run Optuna hyperparameter optimization."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not installed. Run: pip install optuna")

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': seed,
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 50.0),
        }

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        f1_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_set = lgb.Dataset(X_train, y_train)
            val_set = lgb.Dataset(X_val, y_val, reference=train_set)

            model = lgb.train(
                params, train_set,
                num_boost_round=500,
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

            val_preds_proba = model.predict(X_val)
            _, best_f1 = find_optimal_threshold(y_val, val_preds_proba)
            f1_scores.append(best_f1)

        return np.mean(f1_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial:")
    print(f"  F1: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    best_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': seed,
    }
    best_params.update(study.best_trial.params)

    return best_params


def main():
    args = parse_args()
    np.random.seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ROCKET + LightGBM Training for TDE Classification")
    print("=" * 70)
    print(f"Kernels per band: {args.num_kernels}")
    print(f"Total ROCKET features: {args.num_kernels * 6 * 2}")
    print(f"Grid points: {args.grid_points}")
    print(f"Domain features: {args.use_domain_features}")
    print(f"Optuna: {args.optuna}")

    # Load data
    print("\nLoading training data...")
    train_log, train_lc = load_train_data()
    y = train_log['target'].values
    print(f"Loaded {len(train_log)} training objects")
    print(f"Class distribution: TDE={y.sum()}, non-TDE={len(y) - y.sum()}")

    # Prepare ROCKET features
    if args.load_features:
        print(f"\nLoading pre-computed features from {args.load_features}...")
        cached = np.load(args.load_features)
        X_rocket = cached['X_rocket']
        print(f"Loaded features shape: {X_rocket.shape}")
    else:
        print("\nInterpolating training light curves...")
        X_interp = prepare_interpolated_data(
            train_log, train_lc,
            n_points=args.grid_points,
            desc="Interpolating train"
        )
        print(f"Interpolated data shape: {X_interp.shape}")

        print(f"\nFitting ROCKET with {args.num_kernels} kernels per band...")
        rocket = MultiChannelROCKET(
            num_kernels_per_channel=args.num_kernels,
            seed=args.seed,
        )
        X_rocket = rocket.fit_transform(X_interp)
        print(f"ROCKET features shape: {X_rocket.shape}")

        if args.save_features:
            features_path = checkpoint_dir / 'train_features.npz'
            np.savez_compressed(features_path, X_rocket=X_rocket, y=y)
            joblib.dump(rocket, checkpoint_dir / 'rocket_transformer.pkl')
            print(f"Features saved to {features_path}")

    # Extract and combine domain features if requested
    domain_feature_cols = None
    if args.use_domain_features:
        X_domain, domain_feature_cols = extract_domain_features(
            train_log, train_lc, desc="Extracting training domain features"
        )
        print(f"Domain features shape: {X_domain.shape}")

        # Combine ROCKET and domain features
        X_combined = np.hstack([X_rocket, X_domain])
        print(f"Combined features shape: {X_combined.shape}")
    else:
        X_combined = X_rocket

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # LightGBM parameters
    if args.optuna:
        print(f"\n{'=' * 70}")
        print(f"Running Optuna Hyperparameter Optimization ({args.optuna_trials} trials)")
        print(f"{'=' * 70}")
        params = run_optuna_optimization(
            X_scaled, y,
            n_folds=args.n_folds,
            n_trials=args.optuna_trials,
            seed=args.seed,
        )
        with open(checkpoint_dir / 'optuna_best_params.yaml', 'w') as f:
            yaml.dump(params, f)
    else:
        pos_weight = (len(y) - y.sum()) / y.sum()
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': args.num_leaves,
            'learning_rate': args.learning_rate,
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
        print(f"  Best Iteration: {model.best_iteration}")

        fold_results.append({
            'fold': fold,
            'f1': best_f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'threshold': threshold,
            'best_iteration': model.best_iteration,
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
    }

    print(f"F1: {summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}")
    print(f"Precision: {summary['precision_mean']:.4f}")
    print(f"Recall: {summary['recall_mean']:.4f}")
    print(f"ROC-AUC: {summary['roc_auc_mean']:.4f}")
    print(f"Average Threshold: {summary['avg_threshold']:.3f}")

    # Save summary and scaler
    with open(checkpoint_dir / 'cv_summary.yaml', 'w') as f:
        yaml.dump({
            'args': vars(args),
            'params': params,
            'results': fold_results,
            'summary': summary,
            'domain_feature_cols': domain_feature_cols,
            'n_domain_features': len(domain_feature_cols) if domain_feature_cols else 0,
        }, f)

    joblib.dump(scaler, checkpoint_dir / 'scaler.pkl')

    # Feature importance
    print(f"\n{'=' * 70}")
    print("Top 20 Feature Importances (by gain)")
    print(f"{'=' * 70}")

    importance = np.zeros(X_scaled.shape[1])
    for model in models:
        importance += model.feature_importance(importance_type='gain')
    importance /= len(models)

    # Determine feature names
    n_rocket_features = args.num_kernels * 6 * 2
    top_idx = np.argsort(importance)[-20:][::-1]
    for rank, idx in enumerate(top_idx, 1):
        if args.use_domain_features and idx >= n_rocket_features:
            # This is a domain feature
            domain_idx = idx - n_rocket_features
            feat_name = domain_feature_cols[domain_idx] if domain_feature_cols and domain_idx < len(domain_feature_cols) else f"domain_{domain_idx}"
            print(f"  {rank:2d}. {feat_name}: {importance[idx]:.2f}")
        else:
            print(f"  {rank:2d}. ROCKET_{idx:05d}: {importance[idx]:.2f}")

    # If using domain features, show top domain feature importances
    if args.use_domain_features and domain_feature_cols:
        print(f"\n{'=' * 70}")
        print("Top 10 Domain Feature Importances")
        print(f"{'=' * 70}")
        domain_importance = importance[n_rocket_features:]
        top_domain_idx = np.argsort(domain_importance)[-10:][::-1]
        for rank, idx in enumerate(top_domain_idx, 1):
            feat_name = domain_feature_cols[idx] if idx < len(domain_feature_cols) else f"domain_{idx}"
            print(f"  {rank:2d}. {feat_name}: {domain_importance[idx]:.2f}")

    # Feature selection: retrain with only top features
    selected_feature_indices = None
    if args.feature_selection:
        print(f"\n{'=' * 70}")
        print(f"Feature Selection: Keeping Top {args.top_features} Features")
        print(f"{'=' * 70}")

        # Select top features by importance
        selected_feature_indices = np.argsort(importance)[-args.top_features:]
        X_selected = X_scaled[:, selected_feature_indices]
        print(f"Selected features shape: {X_selected.shape}")

        # Save feature indices
        np.save(checkpoint_dir / 'selected_feature_indices.npy', selected_feature_indices)

        # Retrain with selected features
        print("\nRetraining with selected features...")
        oof_preds_selected = np.zeros(len(y))
        fold_results_selected = []
        models_selected = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y)):
            print(f"\n--- Fold {fold} (feature-selected) ---")

            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_set = lgb.Dataset(X_train, y_train)
            val_set = lgb.Dataset(X_val, y_val, reference=train_set)

            model_sel = lgb.train(
                params, train_set,
                num_boost_round=args.num_boost_round,
                valid_sets=[train_set, val_set],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(args.early_stopping_rounds),
                    lgb.log_evaluation(period=100),
                ],
            )

            val_preds_proba = model_sel.predict(X_val)
            oof_preds_selected[val_idx] = val_preds_proba

            threshold, best_f1 = find_optimal_threshold(y_val, val_preds_proba)
            val_preds = (val_preds_proba >= threshold).astype(int)

            precision = precision_score(y_val, val_preds)
            recall = recall_score(y_val, val_preds)
            roc_auc = roc_auc_score(y_val, val_preds_proba)

            print(f"Fold {fold} (selected): F1={best_f1:.4f}, P={precision:.4f}, R={recall:.4f}")

            fold_results_selected.append({
                'fold': fold,
                'f1': best_f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'threshold': threshold,
            })

            # Save fold model
            fold_dir = checkpoint_dir / f'fold_{fold}_selected'
            fold_dir.mkdir(exist_ok=True)
            model_sel.save_model(str(fold_dir / 'model.txt'))
            models_selected.append(model_sel)

        # Summary for selected features
        f1_selected = [r['f1'] for r in fold_results_selected]
        avg_threshold_selected = np.mean([r['threshold'] for r in fold_results_selected])

        print(f"\n{'=' * 70}")
        print("Feature Selection Results")
        print(f"{'=' * 70}")
        print(f"F1 (all features): {summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}")
        print(f"F1 (top {args.top_features}): {np.mean(f1_selected):.4f} +/- {np.std(f1_selected):.4f}")

        # Use selected models if they perform better
        if np.mean(f1_selected) >= summary['f1_mean'] - 0.01:  # Allow 1% tolerance
            print("Using feature-selected models (performance maintained)")
            models = models_selected
            avg_threshold = avg_threshold_selected
            summary['f1_selected'] = float(np.mean(f1_selected))
            summary['f1_selected_std'] = float(np.std(f1_selected))
        else:
            print("Keeping all features (feature selection reduced performance)")
            selected_feature_indices = None

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

    # Load saved ROCKET transformer if we loaded features
    if args.load_features:
        rocket = joblib.load(Path(args.load_features).parent / 'rocket_transformer.pkl')

    print("Applying ROCKET transform...")
    X_test_rocket = rocket.transform(X_test_interp)

    # Extract and combine domain features for test data if used in training
    if args.use_domain_features:
        X_test_domain, _ = extract_domain_features(
            test_log, test_lc, desc="Extracting test domain features"
        )
        X_test_combined = np.hstack([X_test_rocket, X_test_domain])
        print(f"Test combined features shape: {X_test_combined.shape}")
    else:
        X_test_combined = X_test_rocket

    X_test_scaled = scaler.transform(X_test_combined)

    # Apply feature selection if used
    if selected_feature_indices is not None:
        X_test_scaled = X_test_scaled[:, selected_feature_indices]
        print(f"Applied feature selection: {X_test_scaled.shape}")

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

    # Also save probabilities
    probs_df = pd.DataFrame({
        'object_id': test_log['object_id'],
        'probability': test_preds_proba,
    })
    probs_path = args.output.replace('.csv', '_probs.csv')
    probs_df.to_csv(probs_path, index=False)
    print(f"Probabilities saved to: {probs_path}")

    print(f"\n{'=' * 70}")
    print("Training Complete!")
    print(f"{'=' * 70}")
    f1_str = f"{summary['f1_mean']:.2f}"
    domain_str = "+domain" if args.use_domain_features else ""
    print(f"\nTo submit to Kaggle:")
    print(f"  source venv/bin/activate && export KAGGLE_API_TOKEN=KGAT_209f2e6f5fd3bc6c81588500c431b21c && \\")
    print(f"  kaggle competitions submit -c mallorn-astronomical-classification-challenge \\")
    print(f"    -f {args.output} -m 'ROCKET+LightGBM {args.num_kernels}k{domain_str} CV F1={f1_str}'")


if __name__ == '__main__':
    main()
