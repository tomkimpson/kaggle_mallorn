#!/usr/bin/env python3
"""
Training script for MALLORN light curve classification.

Usage:
    python scripts/train.py --model cnn --epochs 50
    python scripts/train.py --model transformer --epochs 100 --batch_size 64
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_train_data
from src.data.dataset import LightCurveDataset, create_data_loaders
from src.models import LightCurveCNN, LightCurveTransformer
from src.models.cnn import LightCurveCNNWithAttention
from src.training import FocalLoss, Trainer
from src.training.metrics import print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train MALLORN classifier')

    # Model
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'cnn_attn', 'transformer'],
                        help='Model architecture')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')

    # Loss
    parser.add_argument('--loss', type=str, default='focal',
                        choices=['focal', 'bce'],
                        help='Loss function')
    parser.add_argument('--focal_alpha', type=float, default=0.75,
                        help='Focal loss alpha (weight for positive class)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma')

    # Data
    parser.add_argument('--max_seq_len', type=int, default=200,
                        help='Maximum sequence length per band')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to train (None = all folds)')

    # System
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda/mps)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')

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


def create_model(args, device: torch.device) -> nn.Module:
    """Create the model based on arguments."""
    if args.model == 'cnn':
        model = LightCurveCNN(
            in_features=3,
            hidden_channels=64,
            band_embedding_dim=128,
            metadata_dim=2,
            classifier_hidden=128,
            dropout=0.3,
            use_metadata=True,
        )
    elif args.model == 'cnn_attn':
        model = LightCurveCNNWithAttention(
            in_features=3,
            hidden_channels=64,
            band_embedding_dim=128,
            metadata_dim=2,
            classifier_hidden=128,
            dropout=0.3,
            use_metadata=True,
        )
    elif args.model == 'transformer':
        model = LightCurveTransformer(
            in_features=3,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dim_feedforward=256,
            metadata_dim=2,
            classifier_hidden=128,
            dropout=0.1,
            max_seq_len=args.max_seq_len,
            use_metadata=True,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model.to(device)


def create_loss(args) -> nn.Module:
    """Create the loss function."""
    if args.loss == 'focal':
        return FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        # Weighted BCE with pos_weight for class imbalance
        # TDE:non-TDE ratio is about 1:20
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]))


def train_fold(
    args,
    train_log,
    train_lc,
    train_idx,
    val_idx,
    fold: int,
    device: torch.device,
) -> dict:
    """Train a single fold."""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold + 1}/{args.n_folds}")
    print(f"{'='*60}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        log_df=train_log,
        lightcurves_df=train_lc,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        use_weighted_sampler=True,
    )

    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    print(f"Train TDEs: {train_log.iloc[train_idx]['target'].sum()}, "
          f"Val TDEs: {train_log.iloc[val_idx]['target'].sum()}")

    # Create model
    model = create_model(args, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # Create loss
    criterion = create_loss(args)
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)

    # Create trainer
    checkpoint_dir = Path(args.checkpoint_dir) / f"fold_{fold}"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        use_amp=device.type == 'cuda',
        gradient_clip=1.0,
        checkpoint_dir=checkpoint_dir,
    )

    # Train
    best_metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_best=True,
    )

    print(f"\nFold {fold + 1} Best Results:")
    if best_metrics:
        print_metrics(best_metrics, prefix="  ")
    else:
        print("  No valid metrics (check for NaN issues)")

    return {
        'fold': fold,
        'metrics': best_metrics if best_metrics else {'f1': 0, 'precision': 0, 'recall': 0},
        'threshold': trainer.best_threshold,
    }


def main():
    args = parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_log, train_lc = load_train_data()
    print(f"Loaded {len(train_log)} objects, {len(train_lc)} observations")
    print(f"TDE count: {train_log['target'].sum()}")

    # Cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    targets = train_log['target'].values

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_log, targets)):
        # Skip if training specific fold
        if args.fold is not None and fold != args.fold:
            continue

        result = train_fold(
            args=args,
            train_log=train_log,
            train_lc=train_lc,
            train_idx=train_idx,
            val_idx=val_idx,
            fold=fold,
            device=device,
        )
        fold_results.append(result)

    # Summary
    if len(fold_results) > 1:
        print(f"\n{'='*60}")
        print("Cross-Validation Summary")
        print(f"{'='*60}")

        f1_scores = [r['metrics']['f1'] for r in fold_results]
        precision_scores = [r['metrics']['precision'] for r in fold_results]
        recall_scores = [r['metrics']['recall'] for r in fold_results]
        thresholds = [r['threshold'] for r in fold_results]

        print(f"F1 Score: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
        print(f"Precision: {np.mean(precision_scores):.4f} +/- {np.std(precision_scores):.4f}")
        print(f"Recall: {np.mean(recall_scores):.4f} +/- {np.std(recall_scores):.4f}")
        print(f"Avg Threshold: {np.mean(thresholds):.3f}")

        # Save summary
        summary = {
            'args': vars(args),
            'results': fold_results,
            'summary': {
                'f1_mean': float(np.mean(f1_scores)),
                'f1_std': float(np.std(f1_scores)),
                'precision_mean': float(np.mean(precision_scores)),
                'recall_mean': float(np.mean(recall_scores)),
                'avg_threshold': float(np.mean(thresholds)),
            }
        }

        summary_path = Path(args.checkpoint_dir) / 'cv_summary.yaml'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
