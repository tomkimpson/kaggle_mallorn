#!/usr/bin/env python3
"""
Stacking ensemble for MALLORN TDE classification.

Combines predictions from CNN, Transformer, LightGBM, and ROCKET+LightGBM using a meta-learner.
Uses out-of-fold predictions for proper stacking to avoid leakage.

Usage:
    python scripts/ensemble_stack.py --train  # Train the stacking model
    python scripts/ensemble_stack.py --predict  # Generate submission
    python scripts/ensemble_stack.py --train --rocket_lgbm_dir checkpoints/rocket_domain_5k
"""

import argparse
import numpy as np
import pandas as pd
import torch
import joblib
import yaml
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from scipy.optimize import minimize
from tqdm import tqdm
import lightgbm as lgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_train_data, load_test_data
from src.data.dataset import LightCurveDataset, create_test_loader
from src.features.light_curve_features import extract_features_batch, get_feature_columns
from src.features.rocket import MultiChannelROCKET
from src.models import LightCurveCNN, LightCurveTransformer
from src.models.cnn import LightCurveCNNWithAttention

BANDS = ['u', 'g', 'r', 'i', 'z', 'y']


def parse_args():
    parser = argparse.ArgumentParser(description='Stacking ensemble')

    parser.add_argument('--train', action='store_true',
                        help='Train the stacking meta-model')
    parser.add_argument('--predict', action='store_true',
                        help='Generate predictions with trained stacker')

    parser.add_argument('--cnn_dir', type=str, default='checkpoints/cnn',
                        help='Directory with CNN fold models')
    parser.add_argument('--transformer_dir', type=str, default='checkpoints/transformer',
                        help='Directory with Transformer fold models')
    parser.add_argument('--lgbm_dir', type=str, default='checkpoints/lgbm',
                        help='Directory with LightGBM fold models')
    parser.add_argument('--rocket_lgbm_dir', type=str, default='checkpoints/rocket_domain_5k',
                        help='Directory with ROCKET+LightGBM fold models (best model)')
    parser.add_argument('--grid_points', type=int, default=100,
                        help='Number of grid points for ROCKET interpolation')

    parser.add_argument('--output_dir', type=str, default='checkpoints/ensemble',
                        help='Output directory for stacker model')
    parser.add_argument('--submission', type=str, default='submission_ensemble.csv',
                        help='Output submission file')

    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for NN inference')
    parser.add_argument('--max_seq_len', type=int, default=200,
                        help='Max sequence length')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for NN inference')

    # Ensemble method
    parser.add_argument('--use_weighted_avg', action='store_true',
                        help='Use optimized weighted averaging instead of logistic regression')

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


def interpolate_lightcurve(obj_lc: pd.DataFrame, n_points: int = 100) -> np.ndarray:
    """Linear interpolation to regular grid for ROCKET."""
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
    """Prepare interpolated light curve data for ROCKET."""
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


def optimize_weights(predictions_list: list, targets: np.ndarray) -> tuple:
    """
    Optimize ensemble weights to maximize F1 score.

    Args:
        predictions_list: List of prediction arrays from different models
        targets: Ground truth labels

    Returns:
        Tuple of (optimal_weights, best_f1, optimal_threshold)
    """
    n_models = len(predictions_list)

    def objective(weights):
        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weighted average predictions
        ensemble_pred = np.zeros_like(predictions_list[0])
        for w, pred in zip(weights, predictions_list):
            ensemble_pred += w * pred

        # Find best F1 for this weight combination
        _, best_f1 = find_optimal_threshold(targets, ensemble_pred)
        return -best_f1  # Minimize negative F1

    # Initial weights (equal)
    initial_weights = np.ones(n_models) / n_models

    # Bounds: each weight between 0 and 1
    bounds = [(0, 1) for _ in range(n_models)]

    # Constraint: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    optimal_weights = result.x / result.x.sum()

    # Calculate ensemble with optimal weights
    ensemble_pred = np.zeros_like(predictions_list[0])
    for w, pred in zip(optimal_weights, predictions_list):
        ensemble_pred += w * pred

    threshold, best_f1 = find_optimal_threshold(targets, ensemble_pred)

    return optimal_weights, best_f1, threshold, ensemble_pred


@torch.no_grad()
def get_nn_predictions(model, dataset, device, batch_size=64):
    """Get predictions from a neural network model."""
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []
    for batch in loader:
        sequences = batch['sequences'].to(device)
        masks = batch['masks'].to(device)
        metadata = batch.get('metadata')
        if metadata is not None:
            metadata = metadata.to(device)

        logits = model(sequences, masks, metadata)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs).flatten()


def load_cnn_model(checkpoint_path: Path, device: torch.device):
    """Load CNN model from checkpoint."""
    model = LightCurveCNN(
        in_features=6,
        hidden_channels=64,
        band_embedding_dim=128,
        metadata_dim=2,
        classifier_hidden=128,
        dropout=0.3,
        use_metadata=True,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def load_transformer_model(checkpoint_path: Path, device: torch.device, max_seq_len=200):
    """Load Transformer model from checkpoint."""
    model = LightCurveTransformer(
        in_features=6,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
        metadata_dim=2,
        classifier_hidden=128,
        dropout=0.1,
        max_seq_len=max_seq_len,
        use_metadata=True,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def get_oof_predictions(args, train_log, train_lc, device):
    """
    Get out-of-fold predictions from all base models.

    Returns a dictionary with OOF predictions for each model type.
    """
    n_samples = len(train_log)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    targets = train_log['target'].values

    oof_cnn = np.zeros(n_samples)
    oof_transformer = np.zeros(n_samples)
    oof_lgbm = np.zeros(n_samples)
    oof_rocket_lgbm = np.zeros(n_samples)

    # Extract features for LightGBM
    print("Extracting features for LightGBM...")
    train_features = extract_features_batch(train_log, train_lc, verbose=False)
    feature_cols = get_feature_columns()
    X_lgbm = train_features[feature_cols].values.astype(np.float32)
    X_lgbm = np.nan_to_num(X_lgbm, nan=0.0, posinf=0.0, neginf=0.0)

    # Prepare ROCKET features if checkpoint exists
    rocket_lgbm_dir = Path(args.rocket_lgbm_dir)
    X_rocket_scaled = None
    rocket_available = False

    if (rocket_lgbm_dir / 'fold_0' / 'model.txt').exists():
        print("Preparing ROCKET features for ensemble...")

        # Check if pre-computed features exist
        features_path = rocket_lgbm_dir / 'train_features.npz'
        rocket_transformer_path = rocket_lgbm_dir / 'rocket_transformer.pkl'
        scaler_path = rocket_lgbm_dir / 'scaler.pkl'

        if features_path.exists() and scaler_path.exists():
            print("Loading pre-computed ROCKET features...")
            cached = np.load(features_path)
            X_rocket = cached['X_rocket']
            scaler = joblib.load(scaler_path)
            X_rocket_scaled = scaler.transform(X_rocket)
            rocket_available = True
        elif rocket_transformer_path.exists() and scaler_path.exists():
            print("Computing ROCKET features from scratch...")
            X_interp = prepare_interpolated_data(
                train_log, train_lc, n_points=args.grid_points, desc="Interpolating train for ROCKET"
            )
            rocket = joblib.load(rocket_transformer_path)
            X_rocket = rocket.transform(X_interp)
            scaler = joblib.load(scaler_path)
            X_rocket_scaled = scaler.transform(X_rocket)
            rocket_available = True
        else:
            print("Warning: ROCKET checkpoint found but missing transformer/scaler. Skipping.")

    print("Getting out-of-fold predictions...")
    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(train_log, targets), total=args.n_folds)):
        # Create validation dataset for NNs
        val_log = train_log.iloc[val_idx].reset_index(drop=True)
        val_obj_ids = set(val_log['object_id'].tolist())
        val_lc = train_lc[train_lc['object_id'].isin(val_obj_ids)]

        val_dataset = LightCurveDataset(
            log_df=val_log,
            lightcurves_df=val_lc,
            max_seq_len=args.max_seq_len,
            is_test=True,  # Don't need labels for prediction
        )

        # CNN predictions
        cnn_path = Path(args.cnn_dir) / f'fold_{fold}' / 'best_model.pt'
        if cnn_path.exists():
            cnn_model = load_cnn_model(cnn_path, device)
            oof_cnn[val_idx] = get_nn_predictions(cnn_model, val_dataset, device, args.batch_size)
            del cnn_model
            torch.cuda.empty_cache() if device.type == 'cuda' else None

        # Transformer predictions
        trans_path = Path(args.transformer_dir) / f'fold_{fold}' / 'best_model.pt'
        if trans_path.exists():
            trans_model = load_transformer_model(trans_path, device, args.max_seq_len)
            oof_transformer[val_idx] = get_nn_predictions(trans_model, val_dataset, device, args.batch_size)
            del trans_model
            torch.cuda.empty_cache() if device.type == 'cuda' else None

        # LightGBM predictions
        lgbm_path = Path(args.lgbm_dir) / f'fold_{fold}' / 'model.txt'
        if lgbm_path.exists():
            lgbm_model = lgb.Booster(model_file=str(lgbm_path))
            oof_lgbm[val_idx] = lgbm_model.predict(X_lgbm[val_idx])

        # ROCKET+LightGBM predictions (best model)
        if rocket_available and X_rocket_scaled is not None:
            rocket_lgbm_path = rocket_lgbm_dir / f'fold_{fold}' / 'model.txt'
            if rocket_lgbm_path.exists():
                rocket_model = lgb.Booster(model_file=str(rocket_lgbm_path))
                oof_rocket_lgbm[val_idx] = rocket_model.predict(X_rocket_scaled[val_idx])

    return {
        'cnn': oof_cnn,
        'transformer': oof_transformer,
        'lgbm': oof_lgbm,
        'rocket_lgbm': oof_rocket_lgbm,
        'targets': targets,
    }


def get_test_predictions(args, test_log, test_lc, device):
    """
    Get averaged test predictions from all base models.

    Each model type's predictions are averaged across folds.
    """
    n_samples = len(test_log)

    # Extract features for LightGBM
    print("Extracting features for LightGBM...")
    test_features = extract_features_batch(test_log, test_lc, verbose=False)
    feature_cols = get_feature_columns()
    X_lgbm = test_features[feature_cols].values.astype(np.float32)
    X_lgbm = np.nan_to_num(X_lgbm, nan=0.0, posinf=0.0, neginf=0.0)

    # Prepare ROCKET features if checkpoint exists
    rocket_lgbm_dir = Path(args.rocket_lgbm_dir)
    X_rocket_scaled = None
    rocket_available = False

    if (rocket_lgbm_dir / 'fold_0' / 'model.txt').exists():
        print("Preparing test ROCKET features...")
        rocket_transformer_path = rocket_lgbm_dir / 'rocket_transformer.pkl'
        scaler_path = rocket_lgbm_dir / 'scaler.pkl'

        if rocket_transformer_path.exists() and scaler_path.exists():
            X_interp = prepare_interpolated_data(
                test_log, test_lc, n_points=args.grid_points, desc="Interpolating test for ROCKET"
            )
            rocket = joblib.load(rocket_transformer_path)
            X_rocket = rocket.transform(X_interp)
            scaler = joblib.load(scaler_path)
            X_rocket_scaled = scaler.transform(X_rocket)
            rocket_available = True

    # Create test dataset for NNs
    test_dataset = LightCurveDataset(
        log_df=test_log,
        lightcurves_df=test_lc,
        max_seq_len=args.max_seq_len,
        is_test=True,
    )

    test_cnn = np.zeros(n_samples)
    test_transformer = np.zeros(n_samples)
    test_lgbm = np.zeros(n_samples)
    test_rocket_lgbm = np.zeros(n_samples)

    n_cnn = 0
    n_trans = 0
    n_lgbm = 0
    n_rocket = 0

    print("Getting test predictions from all folds...")
    for fold in tqdm(range(args.n_folds)):
        # CNN predictions
        cnn_path = Path(args.cnn_dir) / f'fold_{fold}' / 'best_model.pt'
        if cnn_path.exists():
            cnn_model = load_cnn_model(cnn_path, device)
            test_cnn += get_nn_predictions(cnn_model, test_dataset, device, args.batch_size)
            n_cnn += 1
            del cnn_model
            torch.cuda.empty_cache() if device.type == 'cuda' else None

        # Transformer predictions
        trans_path = Path(args.transformer_dir) / f'fold_{fold}' / 'best_model.pt'
        if trans_path.exists():
            trans_model = load_transformer_model(trans_path, device, args.max_seq_len)
            test_transformer += get_nn_predictions(trans_model, test_dataset, device, args.batch_size)
            n_trans += 1
            del trans_model
            torch.cuda.empty_cache() if device.type == 'cuda' else None

        # LightGBM predictions
        lgbm_path = Path(args.lgbm_dir) / f'fold_{fold}' / 'model.txt'
        if lgbm_path.exists():
            lgbm_model = lgb.Booster(model_file=str(lgbm_path))
            test_lgbm += lgbm_model.predict(X_lgbm)
            n_lgbm += 1

        # ROCKET+LightGBM predictions
        if rocket_available and X_rocket_scaled is not None:
            rocket_lgbm_path = rocket_lgbm_dir / f'fold_{fold}' / 'model.txt'
            if rocket_lgbm_path.exists():
                rocket_model = lgb.Booster(model_file=str(rocket_lgbm_path))
                test_rocket_lgbm += rocket_model.predict(X_rocket_scaled)
                n_rocket += 1

    # Average predictions
    if n_cnn > 0:
        test_cnn /= n_cnn
    if n_trans > 0:
        test_transformer /= n_trans
    if n_lgbm > 0:
        test_lgbm /= n_lgbm
    if n_rocket > 0:
        test_rocket_lgbm /= n_rocket

    return {
        'cnn': test_cnn,
        'transformer': test_transformer,
        'lgbm': test_lgbm,
        'rocket_lgbm': test_rocket_lgbm,
        'object_ids': test_features['object_id'].tolist(),
    }


def train_stacker(args):
    """Train the stacking meta-model using OOF predictions."""
    device = get_device(args.device)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    print("Loading training data...")
    train_log, train_lc = load_train_data()
    print(f"Loaded {len(train_log)} training objects")

    # Get OOF predictions
    oof = get_oof_predictions(args, train_log, train_lc, device)

    # Stack features
    available_models = []
    stack_cols = []

    if np.any(oof['cnn'] != 0):
        available_models.append('cnn')
        stack_cols.append(oof['cnn'])
        print(f"CNN OOF predictions available")

    if np.any(oof['transformer'] != 0):
        available_models.append('transformer')
        stack_cols.append(oof['transformer'])
        print(f"Transformer OOF predictions available")

    if np.any(oof['lgbm'] != 0):
        available_models.append('lgbm')
        stack_cols.append(oof['lgbm'])
        print(f"LightGBM OOF predictions available")

    if np.any(oof['rocket_lgbm'] != 0):
        available_models.append('rocket_lgbm')
        stack_cols.append(oof['rocket_lgbm'])
        print(f"ROCKET+LightGBM OOF predictions available (best single model)")

    if not stack_cols:
        print("ERROR: No base model predictions available!")
        return

    X_stack = np.column_stack(stack_cols)
    y = oof['targets']

    print(f"\nStacking {len(available_models)} models: {available_models}")
    print(f"Stack features shape: {X_stack.shape}")

    if args.use_weighted_avg:
        # Weighted averaging approach
        print("\nOptimizing ensemble weights...")
        optimal_weights, best_f1, threshold, meta_probs = optimize_weights(stack_cols, y)

        meta_preds = (meta_probs >= threshold).astype(int)
        precision = precision_score(y, meta_preds)
        recall = recall_score(y, meta_preds)
        roc_auc = roc_auc_score(y, meta_probs)

        print(f"\n{'=' * 60}")
        print("Weighted Averaging Results (on training data)")
        print(f"{'=' * 60}")
        print(f"F1: {best_f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Optimal Threshold: {threshold:.3f}")
        print(f"Optimal Weights: {dict(zip(available_models, optimal_weights))}")

        # Save
        joblib.dump({
            'method': 'weighted_avg',
            'weights': optimal_weights,
            'available_models': available_models,
            'threshold': threshold,
        }, output_dir / 'stacker.pkl')

        # Summary for weighted avg
        summary = {
            'method': 'weighted_avg',
            'available_models': available_models,
            'f1': float(best_f1),
            'precision': float(precision),
            'recall': float(recall),
            'roc_auc': float(roc_auc),
            'threshold': float(threshold),
            'weights': {k: float(v) for k, v in zip(available_models, optimal_weights)},
        }

    else:
        # Logistic regression meta-learner
        print("\nTraining meta-learner (LogisticRegressionCV)...")
        meta_model = LogisticRegressionCV(
            Cs=np.logspace(-4, 4, 20),
            cv=5,
            scoring='f1',
            class_weight='balanced',
            max_iter=1000,
            random_state=args.seed,
        )
        meta_model.fit(X_stack, y)

        # Evaluate
        meta_probs = meta_model.predict_proba(X_stack)[:, 1]
        threshold, best_f1 = find_optimal_threshold(y, meta_probs)
        meta_preds = (meta_probs >= threshold).astype(int)

        precision = precision_score(y, meta_preds)
        recall = recall_score(y, meta_preds)
        roc_auc = roc_auc_score(y, meta_probs)

        print(f"\n{'=' * 60}")
        print("Stacking Results (on training data - optimistic estimate)")
        print(f"{'=' * 60}")
        print(f"F1: {best_f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Optimal Threshold: {threshold:.3f}")
        print(f"Meta-learner C: {meta_model.C_[0]:.6f}")
        print(f"Meta-learner coefficients: {dict(zip(available_models, meta_model.coef_[0]))}")

        # Save
        joblib.dump({
            'method': 'logistic_regression',
            'meta_model': meta_model,
            'available_models': available_models,
            'threshold': threshold,
        }, output_dir / 'stacker.pkl')

        # Summary for logistic regression
        summary = {
            'method': 'logistic_regression',
            'available_models': available_models,
            'f1': float(best_f1),
            'precision': float(precision),
            'recall': float(recall),
            'roc_auc': float(roc_auc),
            'threshold': float(threshold),
            'coefficients': {k: float(v) for k, v in zip(available_models, meta_model.coef_[0])},
        }

    # Save summary (using the summary from either branch above)
    with open(output_dir / 'stacker_summary.yaml', 'w') as f:
        yaml.dump(summary, f)

    print(f"\nStacker saved to: {output_dir / 'stacker.pkl'}")

    # Also save OOF predictions for analysis
    np.savez(
        output_dir / 'oof_predictions.npz',
        cnn=oof['cnn'],
        transformer=oof['transformer'],
        lgbm=oof['lgbm'],
        rocket_lgbm=oof['rocket_lgbm'],
        targets=oof['targets'],
        meta_probs=meta_probs,
    )
    print(f"OOF predictions saved to: {output_dir / 'oof_predictions.npz'}")


def predict_stacker(args):
    """Generate predictions using the trained stacker."""
    device = get_device(args.device)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)

    # Load stacker
    stacker_path = output_dir / 'stacker.pkl'
    if not stacker_path.exists():
        print(f"ERROR: Stacker not found at {stacker_path}")
        print("Run with --train first to train the stacker.")
        return

    stacker_data = joblib.load(stacker_path)
    available_models = stacker_data['available_models']
    threshold = stacker_data['threshold']
    method = stacker_data.get('method', 'logistic_regression')

    print(f"Loaded stacker with models: {available_models}")
    print(f"Ensemble method: {method}")

    # Load test data
    print("\nLoading test data...")
    test_log, test_lc = load_test_data()
    print(f"Loaded {len(test_log)} test objects")

    # Get test predictions
    test_preds = get_test_predictions(args, test_log, test_lc, device)

    # Stack in same order as training
    stack_cols = []
    for model_name in available_models:
        stack_cols.append(test_preds[model_name])

    X_stack = np.column_stack(stack_cols)
    print(f"Stack features shape: {X_stack.shape}")

    # Predict based on method
    if method == 'weighted_avg':
        weights = stacker_data['weights']
        meta_probs = np.zeros(len(stack_cols[0]))
        for w, pred in zip(weights, stack_cols):
            meta_probs += w * pred
        print(f"Using weighted averaging with weights: {dict(zip(available_models, weights))}")
    else:
        meta_model = stacker_data['meta_model']
        meta_probs = meta_model.predict_proba(X_stack)[:, 1]

    predictions = (meta_probs >= threshold).astype(int)

    print(f"\nUsing threshold: {threshold:.3f}")
    print(f"Predicted TDEs: {predictions.sum()}")
    print(f"TDE rate: {predictions.mean():.2%}")

    # Create submission
    submission = pd.DataFrame({
        'object_id': test_preds['object_ids'],
        'prediction': predictions,
    })

    submission.to_csv(args.submission, index=False)
    print(f"\nSubmission saved to: {args.submission}")
    print(submission.head())

    # Also save probabilities
    probs_df = pd.DataFrame({
        'object_id': test_preds['object_ids'],
        'probability': meta_probs,
        'cnn_prob': test_preds['cnn'],
        'transformer_prob': test_preds['transformer'],
        'lgbm_prob': test_preds['lgbm'],
        'rocket_lgbm_prob': test_preds['rocket_lgbm'],
    })
    probs_path = args.submission.replace('.csv', '_probs.csv')
    probs_df.to_csv(probs_path, index=False)
    print(f"Probabilities saved to: {probs_path}")


def main():
    args = parse_args()

    if args.train:
        train_stacker(args)
    elif args.predict:
        predict_stacker(args)
    else:
        print("Please specify --train or --predict")
        print("  --train: Train the stacking meta-model using OOF predictions")
        print("  --predict: Generate submission using trained stacker")


if __name__ == '__main__':
    main()
