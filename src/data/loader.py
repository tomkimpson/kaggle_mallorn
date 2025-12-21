"""
Data loader for MALLORN Kaggle competition.

Loads light curve data from 20 split directories and joins with metadata.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import warnings


# LSST filter bands
BANDS = ['u', 'g', 'r', 'i', 'z', 'y']

# Effective wavelengths in Angstroms (from SVO Filter Profile Service)
BAND_WAVELENGTHS = {
    'u': 3641,
    'g': 4704,
    'r': 6155,
    'i': 7504,
    'z': 8695,
    'y': 10056
}


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent.parent / 'data'


def load_log_file(split: str = 'train') -> pd.DataFrame:
    """
    Load the log file containing object metadata.

    Args:
        split: 'train' or 'test'

    Returns:
        DataFrame with columns: object_id, Z, Z_err, EBV, SpecType,
                               English Translation, split, target
    """
    data_dir = get_data_dir()
    log_path = data_dir / f'{split}_log.csv'

    df = pd.read_csv(log_path)

    # Ensure consistent dtypes
    df['object_id'] = df['object_id'].astype(str)
    df['Z'] = df['Z'].astype(float)
    df['EBV'] = df['EBV'].astype(float)

    if split == 'train':
        df['target'] = df['target'].astype(int)

    return df


def load_split_lightcurves(split_name: str, data_split: str = 'train') -> pd.DataFrame:
    """
    Load light curves from a single split directory.

    Args:
        split_name: e.g., 'split_01', 'split_02', ...
        data_split: 'train' or 'test'

    Returns:
        DataFrame with columns: object_id, Time (MJD), Flux, Flux_err, Filter
    """
    data_dir = get_data_dir()
    lc_path = data_dir / split_name / f'{data_split}_full_lightcurves.csv'

    if not lc_path.exists():
        warnings.warn(f"Light curve file not found: {lc_path}")
        return pd.DataFrame()

    df = pd.read_csv(lc_path)
    df['object_id'] = df['object_id'].astype(str)

    return df


def load_all_lightcurves(data_split: str = 'train', n_workers: int = 4) -> pd.DataFrame:
    """
    Load all light curves from all 20 split directories.

    Args:
        data_split: 'train' or 'test'
        n_workers: Number of parallel workers for loading

    Returns:
        Combined DataFrame with all light curves
    """
    split_names = [f'split_{i:02d}' for i in range(1, 21)]

    def load_split(split_name):
        return load_split_lightcurves(split_name, data_split)

    # Load in parallel for speed
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        dfs = list(executor.map(load_split, split_names))

    # Combine all DataFrames
    combined = pd.concat(dfs, ignore_index=True)

    return combined


def load_train_data(cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load complete training data (metadata + light curves).

    Args:
        cache: Whether to cache the loaded data (not implemented yet)

    Returns:
        Tuple of (log_df, lightcurves_df)
    """
    log_df = load_log_file('train')
    lightcurves_df = load_all_lightcurves('train')

    return log_df, lightcurves_df


def load_test_data(cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load complete test data (metadata + light curves).

    Args:
        cache: Whether to cache the loaded data (not implemented yet)

    Returns:
        Tuple of (log_df, lightcurves_df)
    """
    log_df = load_log_file('test')
    lightcurves_df = load_all_lightcurves('test')

    return log_df, lightcurves_df


def load_object_lightcurve(
    object_id: str,
    lightcurves_df: pd.DataFrame,
    log_df: pd.DataFrame
) -> Dict[str, np.ndarray]:
    """
    Load light curve data for a single object, split by filter band.

    Args:
        object_id: The object identifier
        lightcurves_df: DataFrame containing all light curves
        log_df: DataFrame containing object metadata

    Returns:
        Dictionary with keys for each band ('u', 'g', 'r', 'i', 'z', 'y'),
        each containing a dict with 'time', 'flux', 'flux_err' arrays.
        Also includes 'metadata' with Z, EBV, etc.
    """
    # Get object's light curve data
    obj_lc = lightcurves_df[lightcurves_df['object_id'] == object_id]

    # Get object's metadata
    obj_meta = log_df[log_df['object_id'] == object_id].iloc[0]

    result = {
        'metadata': {
            'object_id': object_id,
            'Z': float(obj_meta['Z']),
            'EBV': float(obj_meta['EBV']),
        }
    }

    # Add target if available (training data)
    if 'target' in obj_meta:
        result['metadata']['target'] = int(obj_meta['target'])

    if 'SpecType' in obj_meta and pd.notna(obj_meta['SpecType']):
        result['metadata']['SpecType'] = str(obj_meta['SpecType'])

    # Split by band
    for band in BANDS:
        band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        result[band] = {
            'time': band_data['Time (MJD)'].values.astype(np.float64),
            'flux': band_data['Flux'].values.astype(np.float64),
            'flux_err': band_data['Flux_err'].values.astype(np.float64),
        }

    return result


def get_class_distribution(log_df: pd.DataFrame) -> Dict[str, int]:
    """
    Get the distribution of object classes in the dataset.

    Args:
        log_df: DataFrame containing object metadata

    Returns:
        Dictionary mapping class names to counts
    """
    if 'SpecType' in log_df.columns:
        return log_df['SpecType'].value_counts().to_dict()
    return {}


def get_target_distribution(log_df: pd.DataFrame) -> Dict[int, int]:
    """
    Get the distribution of binary targets (TDE vs non-TDE).

    Args:
        log_df: DataFrame containing object metadata

    Returns:
        Dictionary mapping target values to counts
    """
    if 'target' in log_df.columns:
        return log_df['target'].value_counts().to_dict()
    return {}
