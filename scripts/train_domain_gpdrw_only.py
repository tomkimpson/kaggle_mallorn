#!/usr/bin/env python3
"""
Domain + GP/DRW Features Only (No ROCKET) for MALLORN TDE Classification.

This model is intentionally simpler for stacking diversity:
- Uses only domain features (~168 features) + GP/DRW features (~69 features)
- No ROCKET features (faster, different error patterns)
- Provides complementary predictions for ensemble stacking

Usage:
    python scripts/train_domain_gpdrw_only.py \
        --use_domain_features \
        --use_gp_drw_features \
        --checkpoint_dir checkpoints/no_rocket
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
from src.features.light_curve_features import extract_features_batch, get_feature_columns
from src.features.drw_features import (
    extract_gp_drw_features_batch,
    get_gp_drw_feature_columns
)


def parse_args():
    parser = argparse.ArgumentParser(description='Domain + GP/DRW Only (No ROCKET)')

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
    parser.add_argument('--load_gp_drw_features', type=str, default=None,
                        help='Load pre-computed GP/DRW features')
    parser.add_argument('--save_gp_drw_features', action='store_true',
                        help='Save GP/DRW features after extraction')

    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/no_rocket',
                        help='Directory to save models')
    parser.add_argument('--output', type=str, default='submission_no_rocket.csv',
                        help='Output submission file')

    return parser.parse_args()


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
    print("Domain + GP/DRW Features Only (No ROCKET)")
    print("=" * 70)
    print(f"Domain features: {args.use_domain_features}")
    print(f"GP/DRW features: {args.use_gp_drw_features}")
    print(f"GP/DRW bands: {gp_drw_bands}")
    print("Note: This model is for stacking diversity (no ROCKET features)")

    # Load data
    print("\nLoading training data...")
    train_log, train_lc = load_train_data()
    y = train_log['target'].values
    print(f"Loaded {len(train_log)} training objects")
    print(f"Class distribution: TDE={y.sum()}, non-TDE={len(y) - y.sum()}")

    # Combine all feature sets (no ROCKET)
    feature_sets = []
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

    if len(feature_sets) == 0:
        raise ValueError("No features selected! Use --use_domain_features and/or --use_gp_drw_features")

    # Combine all features
    X_combined = np.hstack(feature_sets)
    print(f"\nCombined features shape: {X_combined.shape}")
    print(f"Total features: {X_combined.shape[1]} (no ROCKET)")

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
    print(f"  Domain: {summary['n_domain_features']}")
    print(f"  GP/DRW: {summary['n_gp_drw_features']}")
    print(f"  Total: {summary['n_total_features']} (no ROCKET)")

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

    # Feature importance
    print(f"\n{'=' * 70}")
    print("Top 20 Feature Importances")
    print(f"{'=' * 70}")

    importance = np.zeros(X_scaled.shape[1])
    for model in models:
        importance += model.feature_importance(importance_type='gain')
    importance /= len(models)

    sorted_idx = np.argsort(importance)[::-1]
    for rank, idx in enumerate(sorted_idx[:20], 1):
        if idx < len(all_feature_cols):
            print(f"  {rank:2d}. {all_feature_cols[idx]}: {importance[idx]:.2f}")

    # Generate test predictions
    print(f"\n{'=' * 70}")
    print("Generating Test Predictions")
    print(f"{'=' * 70}")

    print("\nLoading test data...")
    test_log, test_lc = load_test_data()
    print(f"Loaded {len(test_log)} test objects")

    # Combine test features
    test_feature_sets = []

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

    # Save probabilities for stacking
    probs_path = args.output.replace('.csv', '_probs.csv')
    pd.DataFrame({
        'object_id': test_log['object_id'],
        'probability': test_preds_proba,
    }).to_csv(probs_path, index=False)
    print(f"Probabilities saved to: {probs_path}")

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
