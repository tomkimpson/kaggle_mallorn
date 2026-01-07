#!/usr/bin/env python3
"""
OOF Stacking Ensemble for MALLORN TDE Classification.

Combines predictions from multiple base models using logistic regression meta-learner.
Uses out-of-fold (OOF) predictions to avoid leakage.

Base models:
1. v1: rocket_domain_optuna
2. v2: gp_drw_v2
3. no_rocket: domain + GP/DRW only
4. (optional) gp_drw_v2_optuna

Usage:
    python scripts/oof_stacker.py
    python scripts/oof_stacker.py --submit  # Also submit to Kaggle
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import joblib
import subprocess
import yaml


def find_optimal_threshold(y_true, y_pred_proba, thresholds=None):
    """Find threshold that maximizes F1 score."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01)

    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def safe_logit(p, eps=1e-7):
    """Safe logit that clips probabilities to avoid inf."""
    p = np.clip(p, eps, 1 - eps)
    return logit(p)


def parse_args():
    parser = argparse.ArgumentParser(description='OOF Stacking Ensemble')

    parser.add_argument('--models', type=str, nargs='+',
                        default=['gp_drw_v2'],
                        help='Model checkpoint directories to stack')
    parser.add_argument('--output_dir', type=str, default='checkpoints/stacker',
                        help='Output directory')
    parser.add_argument('--output', type=str, default='submission_stacked.csv',
                        help='Output submission file')
    parser.add_argument('--submit', action='store_true',
                        help='Submit to Kaggle')

    return parser.parse_args()


def load_model_predictions(checkpoint_dir, model_name):
    """Load OOF and test predictions from a model checkpoint."""
    checkpoint_path = Path(checkpoint_dir)

    # Load OOF predictions
    oof_path = checkpoint_path / 'oof_predictions.npz'
    if not oof_path.exists():
        print(f"  Warning: No OOF predictions found for {model_name}")
        return None, None, None, None

    oof_data = np.load(oof_path)
    oof_preds = oof_data['oof_preds']
    y_true = oof_data['y_true']

    # Load test predictions
    test_path = checkpoint_path / 'test_predictions.npz'
    if not test_path.exists():
        print(f"  Warning: No test predictions found for {model_name}")
        return oof_preds, y_true, None, None

    test_data = np.load(test_path, allow_pickle=True)
    test_preds = test_data['test_preds_proba']
    object_ids = test_data['object_ids']

    print(f"  {model_name}: OOF={len(oof_preds)}, test={len(test_preds)}")

    return oof_preds, y_true, test_preds, object_ids


def load_probs_from_csv(probs_path, model_name):
    """Load test probabilities from CSV file."""
    if not Path(probs_path).exists():
        print(f"  Warning: No probs CSV found for {model_name}: {probs_path}")
        return None, None

    df = pd.read_csv(probs_path)
    df = df.sort_values('object_id').reset_index(drop=True)

    print(f"  {model_name}: test={len(df)} (from CSV)")
    return df['probability'].values, df['object_id'].values


def submit_to_kaggle(submission_file, message):
    """Submit to Kaggle competition."""
    cmd = f'''source venv/bin/activate && export KAGGLE_API_TOKEN=KGAT_209f2e6f5fd3bc6c81588500c431b21c && kaggle competitions submit -c mallorn-astronomical-classification-challenge -f {submission_file} -m "{message}"'''
    print(f"Submitting: {submission_file}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Success: {result.stdout.strip()}")
    else:
        print(f"  Error: {result.stderr.strip()}")
    return result.returncode == 0


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OOF Stacking Ensemble")
    print("=" * 70)

    # Define available models and their sources
    model_configs = {
        'rocket_domain_optuna': {
            'checkpoint': 'checkpoints/rocket_domain_optuna',
            'probs_csv': 'submission_rocket_domain_optuna_probs.csv',
        },
        'gp_drw_v2': {
            'checkpoint': 'checkpoints/gp_drw_v2',
            'probs_csv': 'submission_gp_drw_v2_probs.csv',
        },
        'no_rocket': {
            'checkpoint': 'checkpoints/no_rocket',
            'probs_csv': 'submission_no_rocket_probs.csv',
        },
        'gp_drw_v2_optuna': {
            'checkpoint': 'checkpoints/gp_drw_v2_optuna',
            'probs_csv': 'submission_gp_drw_v2_optuna_probs.csv',
        },
    }

    # Find available models
    print("\nLoading model predictions...")
    oof_predictions = {}
    test_predictions = {}
    y_true = None
    object_ids = None

    for model_name, config in model_configs.items():
        checkpoint_path = Path(config['checkpoint'])
        probs_path = Path(config['probs_csv'])

        if checkpoint_path.exists():
            oof_preds, y, test_preds, obj_ids = load_model_predictions(
                checkpoint_path, model_name
            )

            if oof_preds is not None:
                oof_predictions[model_name] = oof_preds
                if y_true is None:
                    y_true = y

            if test_preds is not None:
                test_predictions[model_name] = test_preds
                if object_ids is None:
                    object_ids = obj_ids
        elif probs_path.exists():
            # Fall back to CSV
            test_preds, obj_ids = load_probs_from_csv(probs_path, model_name)
            if test_preds is not None:
                test_predictions[model_name] = test_preds
                if object_ids is None:
                    object_ids = obj_ids

    print(f"\nModels with OOF predictions: {list(oof_predictions.keys())}")
    print(f"Models with test predictions: {list(test_predictions.keys())}")

    if len(oof_predictions) < 2:
        print("\nWarning: Need at least 2 models with OOF predictions for stacking.")
        print("Only found:", list(oof_predictions.keys()))

        if len(test_predictions) >= 2:
            print("\nFalling back to simple averaging of test predictions...")
            model_names = list(test_predictions.keys())

            # Simple average
            avg_probs = np.zeros(len(test_predictions[model_names[0]]))
            for name, preds in test_predictions.items():
                avg_probs += preds
            avg_probs /= len(test_predictions)

            # Use mean threshold from available models
            avg_threshold = 0.25
            predictions = (avg_probs >= avg_threshold).astype(int)

            # Save submission
            submission = pd.DataFrame({
                'object_id': object_ids,
                'prediction': predictions,
            })
            submission.to_csv(args.output, index=False)
            print(f"\nSimple average submission saved to: {args.output}")

            if args.submit:
                submit_to_kaggle(args.output, f"Simple average of {len(test_predictions)} models")
            return

        print("Error: Not enough models available for stacking.")
        return

    # Stack OOF predictions
    print("\n" + "=" * 70)
    print("Training Logistic Regression Meta-Learner")
    print("=" * 70)

    model_names = sorted(oof_predictions.keys())
    print(f"Stacking models: {model_names}")

    # Build OOF feature matrix (in logits)
    X_oof = np.column_stack([
        safe_logit(oof_predictions[name]) for name in model_names
    ])
    print(f"OOF feature matrix shape: {X_oof.shape}")

    # Train meta-learner with CV
    print("\nTraining LogisticRegressionCV meta-learner...")
    meta = LogisticRegressionCV(
        cv=5,
        class_weight='balanced',
        max_iter=1000,
        solver='lbfgs',
    )
    meta.fit(X_oof, y_true)

    print(f"Best C: {meta.C_[0]:.4f}")
    print(f"Coefficients (model weights):")
    for name, coef in zip(model_names, meta.coef_[0]):
        print(f"  {name}: {coef:.4f}")

    # Get meta-learner OOF predictions
    meta_oof_probs = meta.predict_proba(X_oof)[:, 1]

    # Find optimal threshold on pooled OOF
    threshold, f1 = find_optimal_threshold(y_true, meta_oof_probs)
    print(f"\nPooled OOF threshold: {threshold:.3f}")
    print(f"Pooled OOF F1: {f1:.4f}")

    # Evaluate meta-learner
    meta_oof_preds = (meta_oof_probs >= threshold).astype(int)
    precision = precision_score(y_true, meta_oof_preds)
    recall = recall_score(y_true, meta_oof_preds)
    roc_auc = roc_auc_score(y_true, meta_oof_probs)

    print(f"\nMeta-learner OOF Performance:")
    print(f"  F1: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

    # Generate test predictions
    print("\n" + "=" * 70)
    print("Generating Stacked Test Predictions")
    print("=" * 70)

    # Check which models have test predictions
    available_test_models = [name for name in model_names if name in test_predictions]
    if len(available_test_models) < len(model_names):
        print(f"Warning: Only {len(available_test_models)}/{len(model_names)} models have test predictions")
        print(f"Available: {available_test_models}")

        # Retrain meta-learner with only available models
        print("\nRetraining meta-learner with available models only...")
        X_oof_available = np.column_stack([
            safe_logit(oof_predictions[name]) for name in available_test_models
        ])
        meta_available = LogisticRegressionCV(
            cv=5,
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
        )
        meta_available.fit(X_oof_available, y_true)

        # Get test predictions
        X_test = np.column_stack([
            safe_logit(test_predictions[name]) for name in available_test_models
        ])
        test_probs = meta_available.predict_proba(X_test)[:, 1]

        # Recalculate threshold
        meta_oof_probs_available = meta_available.predict_proba(X_oof_available)[:, 1]
        threshold, f1 = find_optimal_threshold(y_true, meta_oof_probs_available)
        print(f"Adjusted threshold: {threshold:.3f}, F1: {f1:.4f}")
    else:
        # Build test feature matrix
        X_test = np.column_stack([
            safe_logit(test_predictions[name]) for name in model_names
        ])
        test_probs = meta.predict_proba(X_test)[:, 1]

    test_preds = (test_probs >= threshold).astype(int)

    print(f"\nUsing threshold: {threshold:.3f}")
    print(f"Predicted TDEs: {test_preds.sum()}")
    print(f"TDE rate: {test_preds.mean():.2%}")

    # Create submission
    submission = pd.DataFrame({
        'object_id': object_ids,
        'prediction': test_preds,
    })
    submission.to_csv(args.output, index=False)
    print(f"\nSubmission saved to: {args.output}")

    # Save probabilities
    probs_path = args.output.replace('.csv', '_probs.csv')
    pd.DataFrame({
        'object_id': object_ids,
        'probability': test_probs,
    }).to_csv(probs_path, index=False)
    print(f"Probabilities saved to: {probs_path}")

    # Save meta-learner and summary
    joblib.dump(meta, output_dir / 'meta_learner.pkl')

    summary = {
        'models_stacked': model_names,
        'coefficients': dict(zip(model_names, meta.coef_[0].tolist())),
        'best_C': float(meta.C_[0]),
        'threshold': float(threshold),
        'oof_f1': float(f1),
        'oof_precision': float(precision),
        'oof_recall': float(recall),
        'oof_roc_auc': float(roc_auc),
    }
    with open(output_dir / 'stacker_summary.yaml', 'w') as f:
        yaml.dump(summary, f)
    print(f"Summary saved to: {output_dir / 'stacker_summary.yaml'}")

    # Submit if requested
    if args.submit:
        submit_to_kaggle(args.output, f"OOF Stacker: {', '.join(available_test_models)}")

    print(f"\n{'=' * 70}")
    print("Stacking Complete!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
