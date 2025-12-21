"""
Training loop and utilities for model training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, Optional, Callable, List
from pathlib import Path
import time
from tqdm import tqdm

from .metrics import compute_metrics, find_optimal_threshold


class Trainer:
    """
    Trainer class for model training and evaluation.

    Supports:
    - Mixed precision training
    - Early stopping
    - Learning rate scheduling
    - Metric tracking
    - Model checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_amp: bool = True,
        gradient_clip: Optional[float] = 1.0,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: The neural network model
            optimizer: Optimizer for training
            criterion: Loss function
            device: Device to train on (cpu/cuda)
            scheduler: Optional learning rate scheduler
            use_amp: Whether to use automatic mixed precision
            gradient_clip: Max gradient norm for clipping
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Mixed precision - only for CUDA, not MPS
        self.use_amp = use_amp and device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None

        # Tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_metrics: List[Dict[str, float]] = []
        self.best_f1 = 0.0
        self.best_threshold = 0.5
        self.epochs_without_improvement = 0

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            # Move to device
            sequences = batch['sequences'].to(self.device)
            masks = batch['masks'].to(self.device)
            metadata = batch.get('metadata')
            if metadata is not None:
                metadata = metadata.to(self.device)
            targets = batch['target'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    logits = self.model(sequences, masks, metadata)
                    loss = self.criterion(logits, targets)

                # Backward pass with scaling
                self.scaler.scale(loss).backward()

                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(sequences, masks, metadata)
                loss = self.criterion(logits, targets)

                loss.backward()

                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / n_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        find_threshold: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the model on validation data.

        Args:
            val_loader: Validation data loader
            find_threshold: Whether to find optimal threshold

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_probs = []
        all_targets = []

        pbar = tqdm(val_loader, desc="Evaluating", leave=False)
        for batch in pbar:
            sequences = batch['sequences'].to(self.device)
            masks = batch['masks'].to(self.device)
            metadata = batch.get('metadata')
            if metadata is not None:
                metadata = metadata.to(self.device)
            targets = batch['target'].to(self.device)

            if self.use_amp:
                with autocast():
                    logits = self.model(sequences, masks, metadata)
                    loss = self.criterion(logits, targets)
            else:
                logits = self.model(sequences, masks, metadata)
                loss = self.criterion(logits, targets)

            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

        # Aggregate
        all_probs = np.concatenate(all_probs).flatten()
        all_targets = np.concatenate(all_targets).flatten()
        avg_loss = total_loss / n_batches

        # Find optimal threshold
        if find_threshold:
            threshold, _ = find_optimal_threshold(all_targets, all_probs)
        else:
            threshold = self.best_threshold

        # Compute metrics at threshold
        predictions = (all_probs >= threshold).astype(int)
        metrics = compute_metrics(all_targets, predictions, all_probs)
        metrics['loss'] = avg_loss
        metrics['threshold'] = threshold

        self.val_losses.append(avg_loss)
        self.val_metrics.append(metrics)

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        early_stopping_patience: int = 10,
        save_best: bool = True,
    ) -> Dict[str, float]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            early_stopping_patience: Epochs to wait before early stopping
            save_best: Whether to save the best model

        Returns:
            Best validation metrics
        """
        best_metrics = {}

        for epoch in range(n_epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Evaluate
            val_metrics = self.evaluate(val_loader)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1'])
                else:
                    self.scheduler.step()

            # Check for improvement
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.best_threshold = val_metrics['threshold']
                self.epochs_without_improvement = 0
                best_metrics = val_metrics.copy()

                if save_best and self.checkpoint_dir:
                    self.save_checkpoint('best_model.pt')
            else:
                self.epochs_without_improvement += 1

            # Logging
            epoch_time = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch+1}/{n_epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                f"F1: {val_metrics['f1']:.4f}, P: {val_metrics['precision']:.4f}, "
                f"R: {val_metrics['recall']:.4f}, LR: {lr:.2e}"
            )

            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        return best_metrics

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'best_threshold': self.best_threshold,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        self.best_threshold = checkpoint.get('best_threshold', 0.5)

    @torch.no_grad()
    def predict(
        self,
        data_loader: DataLoader,
        threshold: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate predictions for a dataset.

        Args:
            data_loader: Data loader
            threshold: Classification threshold (uses best if None)

        Returns:
            Tuple of (predictions, probabilities, object_ids)
        """
        if threshold is None:
            threshold = self.best_threshold

        self.model.eval()
        all_probs = []
        all_object_ids = []

        for batch in tqdm(data_loader, desc="Predicting"):
            sequences = batch['sequences'].to(self.device)
            masks = batch['masks'].to(self.device)
            metadata = batch.get('metadata')
            if metadata is not None:
                metadata = metadata.to(self.device)

            if self.use_amp:
                with autocast():
                    logits = self.model(sequences, masks, metadata)
            else:
                logits = self.model(sequences, masks, metadata)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_object_ids.extend(batch['object_id'])

        all_probs = np.concatenate(all_probs).flatten()
        predictions = (all_probs >= threshold).astype(int)

        return predictions, all_probs, all_object_ids
