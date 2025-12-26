#!/usr/bin/env python3
"""
Train LightGBM classifier on extracted light curve features.

This script:
1. Extracts domain-specific features from light curves
2. Trains LightGBM with 5-fold stratified CV
3. Optimizes classification threshold for F1
4. Saves models and generates submission

Usage:
    python scripts/train_gbm.py --n_folds 5 --checkpoint_dir checkpoints/lgbm
"""

import argparse
import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import joblib
import sys

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_train_data, load_test_data
from src.features.light_curve_features import extract_features_batch, get_feature_columns


def parse_args():
    parser = argparse.ArgumentParser(description='Train LightGBM on light curve features')

    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/lgbm',
                        help='Directory to save models')
    parser.add_argument('--num_boost_round', type=int, default=1000,
                        help='Maximum number of boosting rounds')
    parser.add_argument('--early_stopping_rounds', type=int, default=50,
                        help='Early stopping rounds')
    parser.add_argument('--output', type=str, default='submission_lgbm.csv',
                        help='Output submission file')
    parser.add_argument('--optuna', action='store_true',
                        help='Run Optuna hyperparameter optimization')
    parser.add_argument('--optuna_trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--use_gp', action='store_true',
                        help='Use GP interpolation for feature extraction (slower but better)')
    parser.add_argument('--use_fats', action='store_true',
                        help='Use FATS astronomical features (slower, +300 features)')

    return parser.parse_args()


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


def run_optuna_optimization(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    n_trials: int,
    seed: int,
) -> dict:
    """
    Run Optuna hyperparameter optimization for LightGBM.

    Returns the best parameters found.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Run: pip install optuna")

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': seed,
            # Tunable parameters
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 30.0),
        }

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        f1_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_set = lgb.Dataset(X_train, y_train)
            val_set = lgb.Dataset(X_val, y_val, reference=train_set)

            model = lgb.train(
                params,
                train_set,
                num_boost_round=500,
                valid_sets=[val_set],
                valid_names=['valid'],
                callbacks=[
                    lgb.early_stopping(30, verbose=False),
                ],
            )

            val_preds_proba = model.predict(X_val)
            _, best_f1 = find_optimal_threshold(y_val, val_preds_proba)
            f1_scores.append(best_f1)

        return np.mean(f1_scores)

    # Create study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial:")
    print(f"  F1: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    # Return best parameters merged with fixed ones
    best_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': seed,
    }
    best_params.update(study.best_trial.params)

    return best_params


def main():
    args = parse_args()

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    np.random.seed(args.seed)

    print("=" * 60)
    print("LightGBM Training for MALLORN TDE Classification")
    print("=" * 60)

    # Load data
    print("\nLoading training data...")
    train_log, train_lc = load_train_data()
    print(f"Loaded {len(train_log)} training objects")

    # Extract features
    print("\nExtracting features from training data...")
    if args.use_gp:
        print("Using GP interpolation (this may take a while)...")
    if args.use_fats:
        print("Using FATS astronomical features (this may take a while)...")
    train_features = extract_features_batch(
        train_log, train_lc, verbose=True,
        use_gp=args.use_gp, use_fats=args.use_fats
    )

    # Get feature columns
    feature_cols = get_feature_columns(include_gp=args.use_gp, include_fats=args.use_fats)
    # Only use columns that exist in the extracted features
    feature_cols = [c for c in feature_cols if c in train_features.columns]
    X = train_features[feature_cols].values.astype(np.float32)
    y = train_features['target'].values.astype(int)

    # Handle any NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)} (TDE: {y.sum()}/{len(y)})")

    # LightGBM parameters
    if args.optuna:
        print(f"\n{'=' * 60}")
        print(f"Running Optuna Hyperparameter Optimization ({args.optuna_trials} trials)")
        print(f"{'=' * 60}")
        params = run_optuna_optimization(
            X, y,
            n_folds=args.n_folds,
            n_trials=args.optuna_trials,
            seed=args.seed,
        )
        # Save best params
        with open(checkpoint_dir / 'optuna_best_params.yaml', 'w') as f:
            yaml.dump(params, f)
        print(f"\nBest parameters saved to: {checkpoint_dir / 'optuna_best_params.yaml'}")
    else:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'scale_pos_weight': (len(y) - y.sum()) / y.sum(),  # Handle imbalance (corrected)
            'min_child_samples': 10,
            'verbose': -1,
            'seed': args.seed,
        }

    # Cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    oof_preds = np.zeros(len(y))
    fold_results = []
    models = []

    print(f"\n{'=' * 60}")
    print(f"Starting {args.n_folds}-fold Cross-Validation")
    print(f"{'=' * 60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create LightGBM datasets
        train_set = lgb.Dataset(X_train, y_train)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set)

        # Train model
        model = lgb.train(
            params,
            train_set,
            num_boost_round=args.num_boost_round,
            valid_sets=[train_set, val_set],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(args.early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ],
        )

        # Predict on validation set
        val_preds_proba = model.predict(X_val)
        oof_preds[val_idx] = val_preds_proba

        # Find optimal threshold
        threshold, best_f1 = find_optimal_threshold(y_val, val_preds_proba)

        # Compute metrics
        val_preds = (val_preds_proba >= threshold).astype(int)
        precision = precision_score(y_val, val_preds)
        recall = recall_score(y_val, val_preds)
        roc_auc = roc_auc_score(y_val, val_preds_proba)

        print(f"\nFold {fold} Results:")
        print(f"  F1: {best_f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Optimal Threshold: {threshold:.3f}")
        print(f"  Best Iteration: {model.best_iteration}")

        # Save fold results
        fold_results.append({
            'fold': fold,
            'f1': best_f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'threshold': threshold,
            'best_iteration': model.best_iteration,
        })

        # Save model
        fold_dir = checkpoint_dir / f'fold_{fold}'
        fold_dir.mkdir(exist_ok=True)
        model.save_model(str(fold_dir / 'model.txt'))
        joblib.dump({'threshold': threshold, 'feature_cols': feature_cols},
                    fold_dir / 'metadata.pkl')

        models.append(model)

    # Summary
    print(f"\n{'=' * 60}")
    print("Cross-Validation Summary")
    print(f"{'=' * 60}")

    f1_scores = [r['f1'] for r in fold_results]
    thresholds = [r['threshold'] for r in fold_results]

    summary = {
        'f1_mean': float(np.mean(f1_scores)),
        'f1_std': float(np.std(f1_scores)),
        'precision_mean': float(np.mean([r['precision'] for r in fold_results])),
        'recall_mean': float(np.mean([r['recall'] for r in fold_results])),
        'roc_auc_mean': float(np.mean([r['roc_auc'] for r in fold_results])),
        'avg_threshold': float(np.mean(thresholds)),
    }

    print(f"F1: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"Precision: {summary['precision_mean']:.4f}")
    print(f"Recall: {summary['recall_mean']:.4f}")
    print(f"ROC-AUC: {summary['roc_auc_mean']:.4f}")
    print(f"Average Threshold: {summary['avg_threshold']:.3f}")

    # Save summary
    with open(checkpoint_dir / 'cv_summary.yaml', 'w') as f:
        yaml.dump({
            'args': vars(args),
            'params': params,
            'results': fold_results,
            'summary': summary,
        }, f)

    # Feature importance
    print(f"\n{'=' * 60}")
    print("Top 20 Feature Importances")
    print(f"{'=' * 60}")

    importance = np.zeros(len(feature_cols))
    for model in models:
        importance += model.feature_importance(importance_type='gain')
    importance /= len(models)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance,
    }).sort_values('importance', ascending=False)

    print(importance_df.head(20).to_string(index=False))
    importance_df.to_csv(checkpoint_dir / 'feature_importance.csv', index=False)

    # Generate submission
    print(f"\n{'=' * 60}")
    print("Generating Test Predictions")
    print(f"{'=' * 60}")

    print("\nLoading test data...")
    test_log, test_lc = load_test_data()
    print(f"Loaded {len(test_log)} test objects")

    print("\nExtracting features from test data...")
    test_features = extract_features_batch(
        test_log, test_lc, verbose=True,
        use_gp=args.use_gp, use_fats=args.use_fats
    )

    X_test = test_features[feature_cols].values.astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensemble predictions from all folds
    test_preds_proba = np.zeros(len(X_test))
    for model in models:
        test_preds_proba += model.predict(X_test)
    test_preds_proba /= len(models)

    # Apply threshold
    avg_threshold = summary['avg_threshold']
    test_preds = (test_preds_proba >= avg_threshold).astype(int)

    print(f"\nUsing threshold: {avg_threshold:.3f}")
    print(f"Predicted TDEs: {test_preds.sum()}")
    print(f"TDE rate: {test_preds.mean():.2%}")

    # Create submission
    submission = pd.DataFrame({
        'object_id': test_features['object_id'],
        'prediction': test_preds,
    })

    submission.to_csv(args.output, index=False)
    print(f"\nSubmission saved to: {args.output}")
    print(submission.head())

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
