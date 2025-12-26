"""
Transformer-based model for light curve classification.

Uses self-attention to model temporal relationships in the light curves
and cross-attention between bands.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sinusoidal functions.

    Can also encode actual time values (not just positions).
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class TimeEmbedding(nn.Module):
    """
    Learnable embedding for continuous time values.

    Projects time deltas to the embedding space.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.time_proj = nn.Linear(1, d_model)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Tensor of shape (batch, seq_len) containing time values

        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        return self.time_proj(time.unsqueeze(-1))


class BandEmbedding(nn.Module):
    """
    Learnable embedding for photometric bands.
    """

    def __init__(self, n_bands: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(n_bands, d_model)

    def forward(self, band_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            band_idx: Tensor of shape (batch, seq_len) with band indices

        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        return self.embedding(band_idx)


class LightCurveTransformer(nn.Module):
    """
    Transformer-based classifier for multi-band light curves.

    Architecture:
    - Per-band temporal transformers
    - Cross-band fusion with attention
    - Classification head with metadata

    Args:
        in_features: Number of input features per observation (flux, flux_err, time_delta, snr, log_flux, cumulative_time)
        d_model: Transformer model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer encoder layers
        dim_feedforward: Feedforward network dimension
        metadata_dim: Dimension of metadata features
        classifier_hidden: Hidden dimension in classifier
        dropout: Dropout rate
        n_bands: Number of photometric bands
        max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        in_features: int = 6,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        metadata_dim: int = 2,
        classifier_hidden: int = 128,
        dropout: float = 0.1,
        n_bands: int = 6,
        max_seq_len: int = 200,
        use_metadata: bool = True,
    ):
        super().__init__()

        self.n_bands = n_bands
        self.use_metadata = use_metadata
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(in_features, d_model)

        # Band embedding
        self.band_embedding = nn.Embedding(n_bands, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder (shared across bands)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Cross-band attention
        self.cross_band_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_band_norm = nn.LayerNorm(d_model)

        # Classification head input dimension
        classifier_input_dim = d_model * n_bands
        if use_metadata:
            classifier_input_dim += metadata_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode_band(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        band_idx: int,
    ) -> torch.Tensor:
        """
        Encode a single band's sequence.

        Args:
            x: Tensor of shape (batch, seq_len, in_features)
            mask: Boolean tensor of shape (batch, seq_len), True = valid
            band_idx: Index of the band (0-5)

        Returns:
            Tensor of shape (batch, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project input features
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add band embedding
        band_emb = self.band_embedding(
            torch.full((batch_size, seq_len), band_idx, device=x.device, dtype=torch.long)
        )
        x = x + band_emb

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create attention mask (True = ignore)
        attn_mask = ~mask

        # Apply transformer
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)

        # Pool over sequence (masked mean)
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        x_masked = x * mask_expanded
        pooled = x_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)

        return pooled

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
        # Encode each band
        band_embeddings = []
        for i in range(self.n_bands):
            band_seq = sequences[:, i, :, :]  # (batch, seq_len, features)
            band_mask = masks[:, i, :]  # (batch, seq_len)
            embedding = self._encode_band(band_seq, band_mask, i)
            band_embeddings.append(embedding)

        # Stack band embeddings: (batch, n_bands, d_model)
        band_stack = torch.stack(band_embeddings, dim=1)

        # Cross-band attention
        attn_out, _ = self.cross_band_attention(band_stack, band_stack, band_stack)
        band_stack = self.cross_band_norm(band_stack + attn_out)

        # Flatten
        combined = band_stack.flatten(1)  # (batch, n_bands * d_model)

        # Add metadata
        if self.use_metadata and metadata is not None:
            combined = torch.cat([combined, metadata], dim=1)

        # Classify
        logits = self.classifier(combined)

        return logits

    def predict_proba(
        self,
        sequences: torch.Tensor,
        masks: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get predicted probabilities."""
        logits = self.forward(sequences, masks, metadata)
        return torch.sigmoid(logits)


class LightCurveTransformerUnified(nn.Module):
    """
    Unified transformer that processes all bands together.

    Instead of separate per-band encoders, this model concatenates
    all observations from all bands into a single sequence with
    band embeddings.
    """

    def __init__(
        self,
        in_features: int = 6,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        metadata_dim: int = 2,
        classifier_hidden: int = 256,
        dropout: float = 0.1,
        n_bands: int = 6,
        max_seq_len: int = 1200,  # 200 * 6 bands
        use_metadata: bool = True,
    ):
        super().__init__()

        self.n_bands = n_bands
        self.use_metadata = use_metadata
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(in_features, d_model)

        # Band embedding
        self.band_embedding = nn.Embedding(n_bands, d_model)

        # Learnable [CLS] token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len + 1, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        classifier_input = d_model + (metadata_dim if use_metadata else 0)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.GELU(),
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
        batch_size, n_bands, seq_len, in_features = sequences.shape

        # Reshape to (batch, n_bands * seq_len, features)
        x = sequences.view(batch_size, n_bands * seq_len, in_features)

        # Project to d_model
        x = self.input_proj(x)  # (batch, total_seq, d_model)

        # Add band embeddings
        band_idx = torch.arange(n_bands, device=x.device).repeat_interleave(seq_len)
        band_idx = band_idx.unsqueeze(0).expand(batch_size, -1)  # (batch, total_seq)
        x = x + self.band_embedding(band_idx)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create attention mask
        flat_mask = masks.view(batch_size, n_bands * seq_len)
        cls_mask = torch.ones(batch_size, 1, device=x.device, dtype=torch.bool)
        full_mask = torch.cat([cls_mask, flat_mask], dim=1)
        attn_mask = ~full_mask

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Get [CLS] token output
        cls_output = x[:, 0, :]  # (batch, d_model)

        # Add metadata
        if self.use_metadata and metadata is not None:
            cls_output = torch.cat([cls_output, metadata], dim=1)

        # Classify
        logits = self.classifier(cls_output)

        return logits
