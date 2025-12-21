#!/usr/bin/env python3
"""
Generate submission file for MALLORN Kaggle competition.

Usage:
    python scripts/submit.py --checkpoint checkpoints/fold_0/best_model.pt
    python scripts/submit.py --checkpoint_dir checkpoints --ensemble
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_test_data
from src.data.dataset import create_test_loader
from src.models import LightCurveCNN, LightCurveTransformer
from src.models.cnn import LightCurveCNNWithAttention


def parse_args():
    parser = argparse.ArgumentParser(description='Generate submission')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to single model checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory containing fold checkpoints')
    parser.add_argument('--ensemble', action='store_true',
                        help='Ensemble all fold models')

    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'cnn_attn', 'transformer'],
                        help='Model architecture')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold (uses saved if None)')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--max_seq_len', type=int, default=200,
                        help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device')

    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output file path')

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get the appropriate device."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


def create_model(model_type: str, device: torch.device) -> torch.nn.Module:
    """Create model by type."""
    if model_type == 'cnn':
        return LightCurveCNN().to(device)
    elif model_type == 'cnn_attn':
        return LightCurveCNNWithAttention().to(device)
    elif model_type == 'transformer':
        return LightCurveTransformer().to(device)
    else:
        raise ValueError(f"Unknown model: {model_type}")


def load_model(checkpoint_path: Path, model_type: str, device: torch.device):
    """Load a single model from checkpoint."""
    model = create_model(model_type, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    threshold = checkpoint.get('best_threshold', 0.5)
    return model, threshold


@torch.no_grad()
def predict(model, data_loader, device, use_amp=False):
    """Generate predictions for a single model."""
    model.eval()
    all_probs = []
    all_object_ids = []

    for batch in tqdm(data_loader, desc="Predicting"):
        sequences = batch['sequences'].to(device)
        masks = batch['masks'].to(device)
        metadata = batch.get('metadata')
        if metadata is not None:
            metadata = metadata.to(device)

        if use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                logits = model(sequences, masks, metadata)
        else:
            logits = model(sequences, masks, metadata)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_object_ids.extend(batch['object_id'])

    all_probs = np.concatenate(all_probs).flatten()
    return all_probs, all_object_ids


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load test data
    print("Loading test data...")
    test_log, test_lc = load_test_data()
    print(f"Loaded {len(test_log)} test objects")

    test_loader = create_test_loader(
        log_df=test_log,
        lightcurves_df=test_lc,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    if args.ensemble:
        # Ensemble predictions from all folds
        checkpoint_dir = Path(args.checkpoint_dir)
        fold_dirs = sorted(checkpoint_dir.glob('fold_*'))

        if not fold_dirs:
            print(f"No fold directories found in {checkpoint_dir}")
            return

        all_fold_probs = []
        all_thresholds = []

        for fold_dir in fold_dirs:
            checkpoint_path = fold_dir / 'best_model.pt'
            if not checkpoint_path.exists():
                print(f"Warning: {checkpoint_path} not found, skipping")
                continue

            print(f"\nLoading {checkpoint_path}...")
            model, threshold = load_model(checkpoint_path, args.model, device)
            all_thresholds.append(threshold)

            probs, object_ids = predict(model, test_loader, device)
            all_fold_probs.append(probs)

        # Average predictions
        ensemble_probs = np.mean(all_fold_probs, axis=0)
        avg_threshold = np.mean(all_thresholds)

        print(f"\nEnsembled {len(all_fold_probs)} models")
        print(f"Average threshold: {avg_threshold:.3f}")

        threshold = args.threshold if args.threshold is not None else avg_threshold
        probs = ensemble_probs

    else:
        # Single model prediction
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            checkpoint_path = Path(args.checkpoint_dir) / 'fold_0' / 'best_model.pt'

        print(f"Loading {checkpoint_path}...")
        model, saved_threshold = load_model(checkpoint_path, args.model, device)
        threshold = args.threshold if args.threshold is not None else saved_threshold

        probs, object_ids = predict(model, test_loader, device)

    # Apply threshold
    predictions = (probs >= threshold).astype(int)

    print(f"\nUsing threshold: {threshold:.3f}")
    print(f"Predicted TDEs: {predictions.sum()}")
    print(f"TDE rate: {predictions.mean():.2%}")

    # Create submission
    submission = pd.DataFrame({
        'object_id': object_ids,
        'prediction': predictions,
    })

    # Save
    submission.to_csv(args.output, index=False)
    print(f"\nSubmission saved to: {args.output}")
    print(submission.head())


if __name__ == '__main__':
    main()
