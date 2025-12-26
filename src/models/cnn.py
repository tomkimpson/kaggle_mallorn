"""
1D CNN model for light curve classification.

Uses per-band encoders that process each filter's time series separately,
then fuses the representations for binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class BandEncoder(nn.Module):
    """
    1D CNN encoder for a single photometric band.

    Processes a sequence of (flux, flux_err, time_delta, snr, log_flux, cumulative_time) observations.
    """

    def __init__(
        self,
        in_features: int = 6,
        hidden_channels: int = 64,
        out_features: int = 128,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(in_features, hidden_channels // 2, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_channels // 2)

        self.conv2 = nn.Conv1d(hidden_channels // 2, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        self.conv4 = nn.Conv1d(hidden_channels, out_features, kernel_size, padding=kernel_size // 2)
        self.bn4 = nn.BatchNorm1d(out_features)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, features)
            mask: Optional boolean mask of shape (batch, seq_len)

        Returns:
            Tensor of shape (batch, out_features)
        """
        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)

        # Apply mask by zeroing out padded positions
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, seq_len)
            x = x * mask.float()

        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = F.relu(self.bn4(self.conv4(x)))

        # Global average pooling
        x = self.pool(x).squeeze(-1)  # (batch, out_features)

        return x


class LightCurveCNN(nn.Module):
    """
    1D CNN classifier for multi-band light curves.

    Architecture:
    - 6 parallel BandEncoders (one per LSST filter: u, g, r, i, z, y)
    - Concatenate band representations
    - Optional metadata fusion (redshift, EBV)
    - MLP classification head

    Args:
        in_features: Number of input features per time step (default: 6 for flux, flux_err, time_delta, snr, log_flux, cumulative_time)
        hidden_channels: Number of channels in conv layers
        band_embedding_dim: Output dimension per band encoder
        metadata_dim: Dimension of metadata features (Z, EBV)
        classifier_hidden: Hidden dimension in classifier MLP
        dropout: Dropout rate
        n_bands: Number of photometric bands
    """

    def __init__(
        self,
        in_features: int = 6,
        hidden_channels: int = 64,
        band_embedding_dim: int = 128,
        metadata_dim: int = 2,
        classifier_hidden: int = 128,
        dropout: float = 0.3,
        n_bands: int = 6,
        use_metadata: bool = True,
    ):
        super().__init__()

        self.n_bands = n_bands
        self.use_metadata = use_metadata

        # Create one encoder per band
        self.band_encoders = nn.ModuleList([
            BandEncoder(
                in_features=in_features,
                hidden_channels=hidden_channels,
                out_features=band_embedding_dim,
                dropout=dropout,
            )
            for _ in range(n_bands)
        ])

        # Calculate input dimension for classifier
        classifier_input_dim = band_embedding_dim * n_bands
        if use_metadata:
            classifier_input_dim += metadata_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        sequences: torch.Tensor,
        masks: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sequences: Tensor of shape (batch, n_bands, seq_len, features)
            masks: Boolean tensor of shape (batch, n_bands, seq_len)
            metadata: Optional tensor of shape (batch, metadata_dim)

        Returns:
            Logits tensor of shape (batch, 1)
        """
        # Process each band separately
        band_embeddings = []
        for i, encoder in enumerate(self.band_encoders):
            band_seq = sequences[:, i, :, :]  # (batch, seq_len, features)
            band_mask = masks[:, i, :]  # (batch, seq_len)
            embedding = encoder(band_seq, band_mask)  # (batch, band_embedding_dim)
            band_embeddings.append(embedding)

        # Concatenate all band embeddings
        combined = torch.cat(band_embeddings, dim=1)  # (batch, n_bands * band_embedding_dim)

        # Add metadata if available
        if self.use_metadata and metadata is not None:
            combined = torch.cat([combined, metadata], dim=1)

        # Classification
        logits = self.classifier(combined)

        return logits

    def predict_proba(
        self,
        sequences: torch.Tensor,
        masks: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict class probabilities.

        Returns:
            Probability tensor of shape (batch, 1)
        """
        logits = self.forward(sequences, masks, metadata)
        return torch.sigmoid(logits)


class LightCurveCNNWithAttention(nn.Module):
    """
    1D CNN with cross-band attention for multi-band light curves.

    Adds attention mechanism to learn relationships between different bands.
    """

    def __init__(
        self,
        in_features: int = 6,
        hidden_channels: int = 64,
        band_embedding_dim: int = 128,
        metadata_dim: int = 2,
        classifier_hidden: int = 128,
        dropout: float = 0.3,
        n_bands: int = 6,
        n_heads: int = 4,
        use_metadata: bool = True,
    ):
        super().__init__()

        self.n_bands = n_bands
        self.use_metadata = use_metadata

        # Create one encoder per band
        self.band_encoders = nn.ModuleList([
            BandEncoder(
                in_features=in_features,
                hidden_channels=hidden_channels,
                out_features=band_embedding_dim,
                dropout=dropout,
            )
            for _ in range(n_bands)
        ])

        # Cross-band attention
        self.band_attention = nn.MultiheadAttention(
            embed_dim=band_embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(band_embedding_dim)

        # Calculate input dimension for classifier
        classifier_input_dim = band_embedding_dim * n_bands
        if use_metadata:
            classifier_input_dim += metadata_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden // 2, 1),
        )

    def forward(
        self,
        sequences: torch.Tensor,
        masks: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sequences: Tensor of shape (batch, n_bands, seq_len, features)
            masks: Boolean tensor of shape (batch, n_bands, seq_len)
            metadata: Optional tensor of shape (batch, metadata_dim)

        Returns:
            Logits tensor of shape (batch, 1)
        """
        # Process each band separately
        band_embeddings = []
        for i, encoder in enumerate(self.band_encoders):
            band_seq = sequences[:, i, :, :]  # (batch, seq_len, features)
            band_mask = masks[:, i, :]  # (batch, seq_len)
            embedding = encoder(band_seq, band_mask)  # (batch, band_embedding_dim)
            band_embeddings.append(embedding)

        # Stack band embeddings: (batch, n_bands, band_embedding_dim)
        band_stack = torch.stack(band_embeddings, dim=1)

        # Apply cross-band attention
        attn_out, _ = self.band_attention(band_stack, band_stack, band_stack)
        band_stack = self.attention_norm(band_stack + attn_out)

        # Flatten band embeddings
        combined = band_stack.flatten(1)  # (batch, n_bands * band_embedding_dim)

        # Add metadata if available
        if self.use_metadata and metadata is not None:
            combined = torch.cat([combined, metadata], dim=1)

        # Classification
        logits = self.classifier(combined)

        return logits
