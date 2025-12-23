"""
FATS (Feature Analysis for Time Series) feature extraction.

Uses the feets library to extract ~80 proven astronomical features from light curves.
These include Stetson indices, structure functions, CAR parameters, and more.

Usage:
    from src.features.fats_features import extract_fats_features, FATSExtractor

    extractor = FATSExtractor()
    features = extractor.extract_single_band(time, mag, err, band='g')
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

try:
    import feets
    FEETS_AVAILABLE = True
except ImportError:
    FEETS_AVAILABLE = False

# LSST photometric bands
BANDS = ['u', 'g', 'r', 'i', 'z', 'y']

# Selected FATS features - curated for transient classification
# These are the most useful features that work with single-band data
SELECTED_FATS_FEATURES = [
    # Basic statistics
    'Amplitude',
    'Mean',
    'Std',
    'Skew',
    'SmallKurtosis',
    'MeanVariance',
    'MedianAbsDev',
    'PercentAmplitude',
    'PercentDiffPercentile',
    'Q31',
    'InterPercentileRange',

    # Variability measures
    'Eta',  # Von Neumann ratio
    'EtaE',  # Weighted eta
    'Con',  # Consecutive same-sign changes
    'Rcs',  # Range of cumulative sum
    'ExcessVariance',
    'ReducedChi2',
    'WeightedMean',

    # Shape descriptors
    'Gskew',  # Median-based skewness
    'Cusum',  # Cumulative sum range
    'AndersonDarling',

    # Trend features
    'LinearTrend',
    'LinearTrend_Sigma',
    'LinearFit_Slope',
    'MaxSlope',
    'PairSlopeTrend',

    # Time-domain features
    'Duration',
    'MaxTimeInterval',
    'MinTimeInterval',
    'TimeMean',
    'TimeStd',
    'Autocor_length',

    # Structure function (variability vs timescale)
    'StructureFunction_index_21',
    'StructureFunction_index_31',
    'StructureFunction_index_32',

    # Stetson indices (designed for variable star detection)
    'StetsonK',
    'StetsonK_AC',

    # Percentile-based features
    'MedianBRP',  # Median buffer range percentage
    'MedianAmplitude',

    # CAR model parameters (damped random walk)
    'CAR_mean',
    'CAR_sigma',
    'CAR_tau',

    # Beyond N-sigma features
    'BeyondNStd',
    'WeightedBeyondNStd',

    # Otsu thresholding features (bimodality)
    'OtsuMeanDiff',
    'OtsuStdLower',
    'OtsuStdUpper',
    'OtsuLowerToAllRatio',
]


class FATSExtractor:
    """
    Wrapper for feets library to extract FATS features from light curves.

    Handles edge cases, NaN values, and provides consistent output.
    """

    def __init__(
        self,
        selected_features: Optional[List[str]] = None,
        min_observations: int = 10,
    ):
        """
        Initialize the FATS extractor.

        Args:
            selected_features: List of features to extract (None = use defaults)
            min_observations: Minimum observations required for extraction
        """
        if not FEETS_AVAILABLE:
            raise ImportError("feets is not installed. Run: pip install feets")

        self.selected_features = selected_features or SELECTED_FATS_FEATURES
        self.min_observations = min_observations

        # Create feature space
        # Note: We use data=['time', 'magnitude', 'error'] to get single-band features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.feature_space = feets.FeatureSpace(
                data=['time', 'magnitude', 'error']
            )

        # Get the actual available feature names from feets
        self._available_features = None

    def _get_available_features(self) -> set:
        """Get the set of features that feets can extract."""
        if self._available_features is None:
            # Extract from a dummy light curve to get feature names
            dummy_time = np.linspace(0, 100, 50)
            dummy_mag = np.random.randn(50) * 0.1 + 15
            dummy_err = np.ones(50) * 0.05

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.feature_space.extract(
                    time=dummy_time, magnitude=dummy_mag, error=dummy_err
                )
                df = result.as_frame()
                self._available_features = set(df.columns)

        return self._available_features

    def extract_single_band(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        band: str,
    ) -> Dict[str, float]:
        """
        Extract FATS features from a single band's light curve.

        Args:
            time: Observation times
            flux: Flux values (will be converted to magnitude)
            flux_err: Flux errors
            band: Band name for prefixing features

        Returns:
            Dictionary of feature name -> value
        """
        prefix = f"{band}_fats_"
        features = {}

        # Initialize all features with defaults
        for feat in self.selected_features:
            features[prefix + feat] = 0.0

        # Check minimum observations
        if len(time) < self.min_observations:
            return features

        # Remove NaN/Inf values
        valid = (
            np.isfinite(time) &
            np.isfinite(flux) &
            np.isfinite(flux_err) &
            (flux > 0) &  # Need positive flux for magnitude conversion
            (flux_err > 0)
        )

        if valid.sum() < self.min_observations:
            return features

        time = time[valid]
        flux = flux[valid]
        flux_err = flux_err[valid]

        # Sort by time
        sort_idx = np.argsort(time)
        time = time[sort_idx]
        flux = flux[sort_idx]
        flux_err = flux_err[sort_idx]

        # Convert flux to magnitude (feets expects magnitudes)
        # mag = -2.5 * log10(flux) + zeropoint
        # We use a relative zeropoint since absolute calibration isn't needed
        mag = -2.5 * np.log10(flux + 1e-10) + 25.0

        # Convert flux error to magnitude error
        # d(mag) = 2.5 / ln(10) * (flux_err / flux)
        mag_err = 2.5 / np.log(10) * (flux_err / (flux + 1e-10))

        # Clip extreme values
        mag = np.clip(mag, 10, 30)
        mag_err = np.clip(mag_err, 0.001, 2.0)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.feature_space.extract(
                    time=time,
                    magnitude=mag,
                    error=mag_err,
                )

                df = result.as_frame()

                # Extract selected features
                available = self._get_available_features()
                for feat in self.selected_features:
                    # Handle features that might have multiple outputs (e.g., BeyondNStd)
                    if feat in df.columns:
                        value = df[feat].iloc[0]
                    elif feat + '_1' in df.columns:
                        # Some features like BeyondNStd become BeyondNStd_1, etc.
                        value = df[feat + '_1'].iloc[0]
                    elif 'Beyond1Std' in df.columns and feat == 'BeyondNStd':
                        value = df['Beyond1Std'].iloc[0]
                    elif 'WeightedBeyond1Std' in df.columns and feat == 'WeightedBeyondNStd':
                        value = df['WeightedBeyond1Std'].iloc[0]
                    else:
                        # Try to find a matching column
                        matching = [c for c in df.columns if c.startswith(feat)]
                        if matching:
                            value = df[matching[0]].iloc[0]
                        else:
                            value = 0.0

                    # Handle NaN/Inf
                    if not np.isfinite(value):
                        value = 0.0

                    features[prefix + feat] = float(value)

        except Exception as e:
            # If extraction fails, return zeros
            pass

        return features

    def extract_all_bands(
        self,
        lightcurve_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Extract FATS features from all bands.

        Args:
            lightcurve_df: DataFrame with columns ['Time (MJD)', 'Flux', 'Flux_err', 'Filter']

        Returns:
            Dictionary of feature name -> value for all bands
        """
        all_features = {}

        for band in BANDS:
            band_data = lightcurve_df[lightcurve_df['Filter'] == band]

            if len(band_data) == 0:
                # Fill with zeros for missing bands
                for feat in self.selected_features:
                    all_features[f"{band}_fats_{feat}"] = 0.0
                continue

            time = band_data['Time (MJD)'].values.astype(np.float64)
            flux = band_data['Flux'].values.astype(np.float64)
            flux_err = band_data['Flux_err'].values.astype(np.float64)

            band_features = self.extract_single_band(time, flux, flux_err, band)
            all_features.update(band_features)

        return all_features


def get_fats_feature_columns() -> List[str]:
    """Get list of all FATS feature column names."""
    columns = []
    for band in BANDS:
        for feat in SELECTED_FATS_FEATURES:
            columns.append(f"{band}_fats_{feat}")
    return columns


def extract_fats_features_batch(
    log_df: pd.DataFrame,
    lightcurves_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract FATS features for all objects.

    Args:
        log_df: DataFrame with object metadata
        lightcurves_df: DataFrame with light curve observations
        verbose: Whether to print progress

    Returns:
        DataFrame with FATS features (one row per object)
    """
    if not FEETS_AVAILABLE:
        raise ImportError("feets is not installed. Run: pip install feets")

    extractor = FATSExtractor()
    all_features = []

    lc_groups = lightcurves_df.groupby('object_id')

    for idx, (_, row) in enumerate(log_df.iterrows()):
        object_id = row['object_id']

        try:
            obj_lc = lc_groups.get_group(object_id)
        except KeyError:
            obj_lc = pd.DataFrame()

        features = {'object_id': object_id}
        fats_features = extractor.extract_all_bands(obj_lc)
        features.update(fats_features)

        if 'target' in row:
            features['target'] = row['target']

        all_features.append(features)

        if verbose and (idx + 1) % 100 == 0:
            print(f"Extracted FATS features for {idx + 1}/{len(log_df)} objects")

    if verbose:
        print(f"FATS feature extraction complete: {len(all_features)} objects")

    return pd.DataFrame(all_features)
