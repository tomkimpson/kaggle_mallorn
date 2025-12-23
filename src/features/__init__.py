"""Feature extraction module for MALLORN light curves."""

from .light_curve_features import (
    extract_per_band_features,
    extract_cross_band_features,
    extract_all_features,
    extract_features_batch,
    get_feature_columns,
)

from .gp_interpolation import (
    GPInterpolator,
    GPResult,
    extract_gp_features,
)

from .fats_features import (
    FATSExtractor,
    extract_fats_features_batch,
    get_fats_feature_columns,
    SELECTED_FATS_FEATURES,
)

__all__ = [
    'extract_per_band_features',
    'extract_cross_band_features',
    'extract_all_features',
    'extract_features_batch',
    'get_feature_columns',
    'GPInterpolator',
    'GPResult',
    'extract_gp_features',
    'FATSExtractor',
    'extract_fats_features_batch',
    'get_fats_feature_columns',
    'SELECTED_FATS_FEATURES',
]
