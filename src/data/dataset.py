"""
PyTorch Dataset classes for MALLORN light curves.

Provides efficient data loading and preprocessing for training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from pathlib import Path

from .loader import BANDS, load_object_lightcurve
from .preprocessing import (
    apply_extinction_correction,
    normalize_time,
    normalize_flux,
    compute_snr
)


class LightCurveDataset(Dataset):
    """
    PyTorch Dataset for MALLORN light curves.

    Each sample contains:
    - Light curve sequences for all 6 bands (padded to fixed length)
    - Metadata (redshift, EBV)
    - Target label (for training data)
    """

    def __init__(
        self,
        log_df: pd.DataFrame,
        lightcurves_df: pd.DataFrame,
        max_seq_len: int = 200,
        apply_extinction: bool = True,
        normalize: bool = True,
        include_metadata: bool = True,
        is_test: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            log_df: DataFrame with object metadata
            lightcurves_df: DataFrame with all light curves
            max_seq_len: Maximum sequence length per band (will pad/truncate)
            apply_extinction: Whether to apply extinction correction
            normalize: Whether to normalize flux values
            include_metadata: Whether to include Z, EBV as features
            is_test: Whether this is test data (no labels)
        """
        self.log_df = log_df
        self.lightcurves_df = lightcurves_df
        self.max_seq_len = max_seq_len
        self.apply_extinction = apply_extinction
        self.normalize = normalize
        self.include_metadata = include_metadata
        self.is_test = is_test

        # Get list of object IDs
        self.object_ids = log_df['object_id'].tolist()

        # Create index for fast lookup
        self.lc_index = lightcurves_df.groupby('object_id').groups

        # Pre-compute class weights for imbalanced sampling
        if not is_test and 'target' in log_df.columns:
            targets = log_df['target'].values
            class_counts = np.bincount(targets)
            self.class_weights = 1.0 / class_counts
            self.sample_weights = self.class_weights[targets]
        else:
            self.class_weights = None
            self.sample_weights = None

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
            - 'sequences': Tensor of shape (6, max_seq_len, 3) for all bands
                          Last dim is [flux, flux_err, time_delta]
            - 'masks': Tensor of shape (6, max_seq_len) indicating valid positions
            - 'lengths': Tensor of shape (6,) with actual sequence lengths
            - 'metadata': Tensor of shape (2,) with [Z, EBV] if include_metadata
            - 'target': Tensor of shape (1,) with label if not is_test
            - 'object_id': String object identifier
        """
        object_id = self.object_ids[idx]

        # Get metadata
        meta_row = self.log_df[self.log_df['object_id'] == object_id].iloc[0]
        z = float(meta_row['Z'])
        ebv = float(meta_row['EBV'])

        # Get light curve data for this object
        obj_lc = self.lightcurves_df.loc[
            self.lightcurves_df['object_id'] == object_id
        ]

        # Process each band
        sequences = []
        masks = []
        lengths = []

        for band in BANDS:
            band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

            time = band_data['Time (MJD)'].values.astype(np.float32)
            flux = band_data['Flux'].values.astype(np.float32)
            flux_err = band_data['Flux_err'].values.astype(np.float32)

            # Remove NaN values
            valid_mask = ~(np.isnan(time) | np.isnan(flux) | np.isnan(flux_err))
            time = time[valid_mask]
            flux = flux[valid_mask]
            flux_err = flux_err[valid_mask]

            # Apply extinction correction
            if self.apply_extinction and len(flux) > 0:
                flux, flux_err = self._apply_extinction(flux, flux_err, ebv, band)

            # Normalize time to start from 0
            if len(time) > 0:
                time = time - time[0]

            # Normalize flux
            if self.normalize and len(flux) > 0:
                scale = np.max(np.abs(flux)) if np.max(np.abs(flux)) > 0 else 1.0
                flux = flux / scale
                flux_err = flux_err / scale

            # Replace any remaining NaN/Inf with 0
            flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
            flux_err = np.nan_to_num(flux_err, nan=0.0, posinf=0.0, neginf=0.0)
            time = np.nan_to_num(time, nan=0.0, posinf=0.0, neginf=0.0)

            # Pad or truncate to max_seq_len
            seq, mask, length = self._pad_sequence(time, flux, flux_err)

            sequences.append(seq)
            masks.append(mask)
            lengths.append(length)

        # Stack all bands
        sequences = np.stack(sequences, axis=0)  # (6, max_seq_len, 3)
        masks = np.stack(masks, axis=0)  # (6, max_seq_len)
        lengths = np.array(lengths)  # (6,)

        result = {
            'sequences': torch.from_numpy(sequences).float(),
            'masks': torch.from_numpy(masks).bool(),
            'lengths': torch.from_numpy(lengths).long(),
            'object_id': object_id,
        }

        # Add metadata features
        if self.include_metadata:
            metadata = np.array([z, ebv], dtype=np.float32)
            result['metadata'] = torch.from_numpy(metadata)

        # Add target label
        if not self.is_test and 'target' in meta_row:
            target = np.array([meta_row['target']], dtype=np.float32)
            result['target'] = torch.from_numpy(target)

        return result

    def _apply_extinction(
        self,
        flux: np.ndarray,
        flux_err: np.ndarray,
        ebv: float,
        band: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply extinction correction using Fitzpatrick99."""
        try:
            from extinction import fitzpatrick99

            wavelengths = {
                'u': 3641.0, 'g': 4704.0, 'r': 6155.0,
                'i': 7504.0, 'z': 8695.0, 'y': 10056.0
            }
            wl = np.array([wavelengths[band]])
            a_lambda = fitzpatrick99(wl, ebv * 3.1)[0]
            factor = 10 ** (a_lambda / 2.5)
            return flux * factor, flux_err * factor
        except ImportError:
            return flux, flux_err

    def _pad_sequence(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Pad or truncate sequence to fixed length.

        Returns:
            Tuple of (sequence, mask, original_length)
        """
        n = len(time)
        seq = np.zeros((self.max_seq_len, 3), dtype=np.float32)
        mask = np.zeros(self.max_seq_len, dtype=np.float32)

        if n == 0:
            return seq, mask, 0

        # Truncate if too long (sample uniformly)
        if n > self.max_seq_len:
            indices = np.linspace(0, n - 1, self.max_seq_len, dtype=int)
            time = time[indices]
            flux = flux[indices]
            flux_err = flux_err[indices]
            n = self.max_seq_len

        # Compute time deltas (time between consecutive observations)
        time_delta = np.zeros_like(time)
        if len(time) > 1:
            time_delta[1:] = np.diff(time)

        # Fill sequence
        seq[:n, 0] = flux
        seq[:n, 1] = flux_err
        seq[:n, 2] = time_delta
        mask[:n] = 1.0

        return seq, mask, n

    def get_sampler(self) -> Optional[WeightedRandomSampler]:
        """
        Get a weighted random sampler for handling class imbalance.

        Returns:
            WeightedRandomSampler or None if test data
        """
        if self.sample_weights is None:
            return None

        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self),
            replacement=True
        )


def create_data_loaders(
    log_df: pd.DataFrame,
    lightcurves_df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 32,
    max_seq_len: int = 200,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.

    Args:
        log_df: DataFrame with object metadata
        lightcurves_df: DataFrame with all light curves
        train_idx: Indices for training data
        val_idx: Indices for validation data
        batch_size: Batch size
        max_seq_len: Maximum sequence length per band
        num_workers: Number of data loading workers
        use_weighted_sampler: Whether to use weighted sampling for imbalance

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split log dataframes
    train_log = log_df.iloc[train_idx].reset_index(drop=True)
    val_log = log_df.iloc[val_idx].reset_index(drop=True)

    # Get object IDs for filtering light curves
    train_obj_ids = set(train_log['object_id'].tolist())
    val_obj_ids = set(val_log['object_id'].tolist())

    train_lc = lightcurves_df[lightcurves_df['object_id'].isin(train_obj_ids)]
    val_lc = lightcurves_df[lightcurves_df['object_id'].isin(val_obj_ids)]

    # Create datasets
    train_dataset = LightCurveDataset(
        log_df=train_log,
        lightcurves_df=train_lc,
        max_seq_len=max_seq_len,
        is_test=False,
    )

    val_dataset = LightCurveDataset(
        log_df=val_log,
        lightcurves_df=val_lc,
        max_seq_len=max_seq_len,
        is_test=False,
    )

    # Create samplers
    train_sampler = train_dataset.get_sampler() if use_weighted_sampler else None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_test_loader(
    log_df: pd.DataFrame,
    lightcurves_df: pd.DataFrame,
    batch_size: int = 32,
    max_seq_len: int = 200,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create test data loader.

    Args:
        log_df: DataFrame with object metadata
        lightcurves_df: DataFrame with all light curves
        batch_size: Batch size
        max_seq_len: Maximum sequence length per band
        num_workers: Number of data loading workers

    Returns:
        Test DataLoader
    """
    test_dataset = LightCurveDataset(
        log_df=log_df,
        lightcurves_df=lightcurves_df,
        max_seq_len=max_seq_len,
        is_test=True,
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
