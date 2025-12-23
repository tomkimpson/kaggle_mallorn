#!/usr/bin/env python3
"""
Generate predictions using trained ROCKET models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_test_data
from src.features.rocket import MultiChannelROCKET

BANDS = ['u', 'g', 'r', 'i', 'z', 'y']


def interpolate_lightcurve_simple(obj_lc, n_points=100):
    """Simple linear interpolation without GP."""
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


def prepare_test_data(log_df, lc_df, n_points=100):
    """Prepare test data."""
    n_samples = len(log_df)
    n_bands = len(BANDS)
    X = np.zeros((n_samples, n_bands, n_points))

    for idx, row in tqdm(log_df.iterrows(), total=n_samples, desc="Interpolating test"):
        obj_id = row['object_id']
        obj_lc = lc_df[lc_df['object_id'] == obj_id]
        X[idx] = interpolate_lightcurve_simple(obj_lc, n_points)

    return X


def main():
    checkpoint_dir = Path('checkpoints/rocket')

    # Load CV summary to get args
    summary = joblib.load(checkpoint_dir / 'cv_summary.pkl')
    args = summary['args']
    avg_threshold = summary['avg_threshold']
    n_folds = args['n_folds']
    grid_points = args['grid_points']

    print("=" * 60)
    print("ROCKET Test Prediction Generation")
    print("=" * 60)
    print(f"Using {n_folds} fold models")
    print(f"Average threshold: {avg_threshold:.3f}")
    print(f"Grid points: {grid_points}")

    # Load test data
    print("\nLoading test data...")
    test_log, test_lc = load_test_data()
    print(f"Loaded {len(test_log)} test objects")

    # Interpolate test data
    print("\nInterpolating test light curves...")
    X_test = prepare_test_data(test_log, test_lc, n_points=grid_points)
    print(f"Test data shape: {X_test.shape}")

    # Load models and generate predictions
    print("\nGenerating ensemble predictions...")
    test_preds_proba = np.zeros(len(X_test))

    for fold in range(n_folds):
        print(f"Predicting with fold {fold} model...")
        fold_dir = checkpoint_dir / f'fold_{fold}'
        model_data = joblib.load(fold_dir / 'model.pkl')

        rocket = model_data['rocket']
        scaler = model_data['scaler']
        clf = model_data['classifier']

        # Transform and predict
        X_test_features = rocket.transform(X_test)
        X_test_scaled = scaler.transform(X_test_features)

        if args['classifier'] == 'ridge':
            decision = clf.decision_function(X_test_scaled)
            proba = 1 / (1 + np.exp(-decision))
        else:
            proba = clf.predict_proba(X_test_scaled)[:, 1]

        test_preds_proba += proba

    # Average predictions
    test_preds_proba /= n_folds
    test_preds = (test_preds_proba >= avg_threshold).astype(int)

    print(f"\nUsing threshold: {avg_threshold:.3f}")
    print(f"Predicted TDEs: {test_preds.sum()}")
    print(f"TDE rate: {test_preds.mean():.2%}")

    # Create submission
    submission = pd.DataFrame({
        'object_id': test_log['object_id'],
        'prediction': test_preds,
    })

    output_file = 'submission_rocket.csv'
    submission.to_csv(output_file, index=False)
    print(f"\nSubmission saved to: {output_file}")

    print(f"\n{'=' * 60}")
    print("Prediction Complete!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
