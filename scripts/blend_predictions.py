#!/usr/bin/env python3
"""
Blend predictions from v1 (rocket_domain_optuna) and v2 (gp_drw_v2) models.

Creates submission files for different blend weights.
Uses v2's OOF predictions to calibrate threshold (since v1 OOF not saved).

Usage:
    python scripts/blend_predictions.py
    python scripts/blend_predictions.py --submit  # Also submit to Kaggle
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
import subprocess


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


def parse_args():
    parser = argparse.ArgumentParser(description='Blend v1 and v2 predictions')

    parser.add_argument('--v1_probs', type=str,
                        default='submission_rocket_domain_optuna_probs.csv',
                        help='v1 probability predictions')
    parser.add_argument('--v2_probs', type=str,
                        default='submission_gp_drw_v2_probs.csv',
                        help='v2 probability predictions')
    parser.add_argument('--v2_oof', type=str,
                        default='checkpoints/gp_drw_v2/oof_predictions.npz',
                        help='v2 OOF predictions for threshold calibration')
    parser.add_argument('--weights', type=str, default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0',
                        help='Comma-separated blend weights for v2')
    parser.add_argument('--output_dir', type=str, default='blends',
                        help='Output directory for blended submissions')
    parser.add_argument('--submit', action='store_true',
                        help='Submit blended predictions to Kaggle')

    return parser.parse_args()


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

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load probability predictions
    print("Loading predictions...")
    v1_df = pd.read_csv(args.v1_probs)
    v2_df = pd.read_csv(args.v2_probs)

    # Ensure same order
    v1_df = v1_df.sort_values('object_id').reset_index(drop=True)
    v2_df = v2_df.sort_values('object_id').reset_index(drop=True)

    # Verify alignment
    assert (v1_df['object_id'] == v2_df['object_id']).all(), "Object IDs must match!"

    v1_probs = v1_df['probability'].values
    v2_probs = v2_df['probability'].values
    object_ids = v1_df['object_id'].values

    print(f"Loaded {len(v1_probs)} test predictions")
    print(f"  v1 mean prob: {v1_probs.mean():.4f}")
    print(f"  v2 mean prob: {v2_probs.mean():.4f}")

    # Load OOF predictions for threshold calibration
    print("\nLoading v2 OOF predictions for threshold calibration...")
    oof_data = np.load(args.v2_oof)
    y_true = oof_data['y_true']
    v2_oof = oof_data['oof_preds']
    v2_threshold = float(oof_data['optimal_threshold'])
    print(f"  v2 OOF samples: {len(y_true)}")
    print(f"  v2 optimal threshold: {v2_threshold:.4f}")

    # Parse weights
    weights = [float(w) for w in args.weights.split(',')]

    print(f"\nGenerating blends for {len(weights)} weight values...")
    print("-" * 60)

    results = []

    for w in weights:
        # Blend test probabilities: p_blend = w*v2 + (1-w)*v1
        blend_probs = w * v2_probs + (1 - w) * v1_probs

        # For threshold calibration, we use v2's OOF (since we don't have v1 OOF)
        # This is an approximation - ideally we'd have both OOF predictions
        # For now, use a simple heuristic: interpolate threshold
        if w == 0:
            # Pure v1 - use a typical threshold (0.3 worked well in Optuna)
            threshold = 0.328  # from v1's cv_summary
        elif w == 1:
            # Pure v2 - use v2's optimal threshold
            threshold = v2_threshold
        else:
            # Interpolate between v1 and v2 thresholds
            v1_threshold = 0.328
            threshold = w * v2_threshold + (1 - w) * v1_threshold

        # Apply threshold
        predictions = (blend_probs >= threshold).astype(int)

        # Count positives
        n_pos = predictions.sum()
        pos_rate = n_pos / len(predictions)

        # Create submission
        submission = pd.DataFrame({
            'object_id': object_ids,
            'prediction': predictions
        })

        # Save
        filename = f'submission_blend_w{w:.1f}.csv'
        filepath = output_dir / filename
        submission.to_csv(filepath, index=False)

        # Also save probabilities
        probs_filename = f'submission_blend_w{w:.1f}_probs.csv'
        probs_filepath = output_dir / probs_filename
        pd.DataFrame({
            'object_id': object_ids,
            'probability': blend_probs
        }).to_csv(probs_filepath, index=False)

        print(f"w={w:.1f}: threshold={threshold:.3f}, positives={n_pos} ({pos_rate:.2%}) -> {filename}")

        results.append({
            'weight': w,
            'threshold': threshold,
            'n_positives': n_pos,
            'pos_rate': pos_rate,
            'file': str(filepath)
        })

        # Submit if requested
        if args.submit:
            submit_to_kaggle(filepath, f"Blend w={w:.1f} (v2={w:.0%}, v1={1-w:.0%})")

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_path = output_dir / 'blend_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("BLEND SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    print(f"\n{len(weights)} submission files created in {output_dir}/")
    if not args.submit:
        print("\nTo submit to Kaggle, run with --submit flag")


if __name__ == '__main__':
    main()
