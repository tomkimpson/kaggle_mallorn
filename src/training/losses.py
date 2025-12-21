"""
Loss functions for imbalanced binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    Reduces the loss contribution from easy-to-classify examples,
    focusing training on hard examples.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'none', 'mean', or 'sum'
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model outputs of shape (batch, 1) or (batch,)
            targets: Binary targets of shape (batch, 1) or (batch,)

        Returns:
            Loss tensor
        """
        # Ensure correct shapes
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Clamp logits to prevent numerical instability
        logits = torch.clamp(logits, min=-50, max=50)

        # Compute probabilities with clamping
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)

        # Compute BCE loss (without reduction)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        p_t = torch.clamp(p_t, min=1e-7, max=1 - 1e-7)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss.

    Applies class weights to handle imbalance.

    Args:
        pos_weight: Weight for positive class (TDE)
    """

    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            logits: Raw model outputs of shape (batch, 1) or (batch,)
            targets: Binary targets of shape (batch, 1) or (batch,)

        Returns:
            Loss tensor
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        if self.pos_weight is not None:
            weight = torch.ones_like(targets)
            weight[targets == 1] = self.pos_weight
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, weight=weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets)

        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for imbalanced classification.

    Different focusing for positive and negative samples.

    Args:
        gamma_neg: Focusing parameter for negative samples
        gamma_pos: Focusing parameter for positive samples
        clip: Probability margin for clipping
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute asymmetric loss.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        probs = torch.sigmoid(logits)

        # Asymmetric clipping
        probs_neg = probs.clamp(min=self.clip)

        # Positive samples
        pos_loss = targets * torch.log(probs.clamp(min=1e-8)) * ((1 - probs) ** self.gamma_pos)

        # Negative samples (with probability shifting)
        neg_loss = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8)) * (probs_neg ** self.gamma_neg)

        loss = -(pos_loss + neg_loss)

        return loss.mean()
