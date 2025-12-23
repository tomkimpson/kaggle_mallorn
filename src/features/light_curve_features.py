"""
Domain-specific feature extraction for MALLORN TDE classification.

Extracts statistical and physical features from light curves that are
known to be useful for TDE identification:
- TDEs have characteristic t^(-5/3) power-law decay
- Specific color evolution (blue to red)
- High luminosity peaks with smooth light curves
- Typical rise times of ~10-50 days
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats

# LSST photometric bands
BANDS = ['u', 'g', 'r', 'i', 'z', 'y']

# TDE canonical power-law decay exponent
TDE_CANONICAL_ALPHA = -5.0 / 3.0  # ≈ -1.667


def fit_power_law_decay(
    time_post_peak: np.ndarray,
    flux_post_peak: np.ndarray,
    flux_err_post_peak: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Fit power-law decay F = A * t^alpha with goodness-of-fit.

    Args:
        time_post_peak: Time since peak (must be > 0)
        flux_post_peak: Flux values post-peak
        flux_err_post_peak: Flux errors post-peak

    Returns:
        Tuple of (alpha, chi2_reduced, alpha_deviation_from_tde)
    """
    if len(time_post_peak) < 3:
        return 0.0, np.inf, np.inf

    # Filter to valid (positive) values
    valid = (time_post_peak > 0) & (flux_post_peak > 0) & (flux_err_post_peak > 0)
    if np.sum(valid) < 3:
        return 0.0, np.inf, np.inf

    t = time_post_peak[valid]
    f = flux_post_peak[valid]
    f_err = flux_err_post_peak[valid]

    # Log-log linear fit: log(F) = log(A) + alpha * log(t)
    log_t = np.log(t)
    log_f = np.log(f)

    try:
        # Weighted fit using flux errors
        weights = 1.0 / (f_err / f + 1e-8)  # Weight by relative error
        weights /= weights.sum()

        coeffs = np.polyfit(log_t, log_f, 1, w=weights)
        alpha = coeffs[0]
        log_A = coeffs[1]

        # Predicted log-flux
        log_f_pred = alpha * log_t + log_A

        # Chi-squared (in log space, weighted)
        residuals = log_f - log_f_pred
        chi2 = np.sum(weights * residuals**2)
        chi2_reduced = chi2 / (len(t) - 2) if len(t) > 2 else chi2

        # Deviation from TDE canonical decay
        alpha_deviation = abs(alpha - TDE_CANONICAL_ALPHA)

        return alpha, chi2_reduced, alpha_deviation

    except (np.linalg.LinAlgError, ValueError):
        return 0.0, np.inf, np.inf


def compute_smoothness_features(
    flux: np.ndarray,
    time: np.ndarray,
) -> Dict[str, float]:
    """
    Compute smoothness features - TDEs have smooth light curves, SNe have bumps.

    Args:
        flux: Flux values
        time: Time values

    Returns:
        Dictionary with smoothness features
    """
    features = {}

    if len(flux) < 4:
        return {
            'n_inflection_points': 0.0,
            'max_acceleration': 0.0,
            'monotonic_ratio': 0.0,
            'roughness': 0.0,
        }

    # First derivative (velocity)
    dt = np.diff(time)
    dt[dt == 0] = 1e-8  # Avoid division by zero
    df = np.diff(flux)
    velocity = df / dt

    # Second derivative (acceleration)
    if len(velocity) > 1:
        dt2 = (dt[:-1] + dt[1:]) / 2
        d_velocity = np.diff(velocity)
        acceleration = d_velocity / dt2

        # Number of inflection points (sign changes in acceleration)
        sign_changes = np.diff(np.sign(acceleration))
        features['n_inflection_points'] = float(np.sum(sign_changes != 0))

        # Maximum absolute acceleration (normalized by flux range)
        flux_range = np.max(flux) - np.min(flux) + 1e-8
        features['max_acceleration'] = float(np.max(np.abs(acceleration)) / flux_range)
    else:
        features['n_inflection_points'] = 0.0
        features['max_acceleration'] = 0.0

    # Monotonic ratio (fraction of consistent direction)
    increasing = np.sum(df > 0)
    decreasing = len(df) - increasing
    features['monotonic_ratio'] = float(max(increasing, decreasing) / len(df))

    # Roughness: std of second derivative normalized
    if len(velocity) > 1:
        features['roughness'] = float(np.std(acceleration) / (np.mean(np.abs(acceleration)) + 1e-8))
    else:
        features['roughness'] = 0.0

    return features


def extract_per_band_features(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    band: str,
) -> Dict[str, float]:
    """
    Extract statistical and physical features from a single band's light curve.

    Args:
        time: Array of observation times (MJD)
        flux: Array of flux values
        flux_err: Array of flux errors
        band: Band name (u, g, r, i, z, y)

    Returns:
        Dictionary of feature name -> value
    """
    prefix = f"{band}_"
    features = {}

    # Handle empty or very short light curves
    if len(flux) < 3:
        # Return zeros for all features
        feature_names = [
            'peak_flux', 'peak_time', 'mean_flux', 'std_flux', 'median_flux',
            'skewness', 'kurtosis', 'rise_time', 'decay_rate', 'amplitude_ratio',
            'duration', 'above_baseline_frac', 'snr_mean', 'snr_max',
            # New power-law features
            'decay_chi2', 'decay_alpha_dev',
            # Smoothness features
            'n_inflection_points', 'max_acceleration', 'monotonic_ratio', 'roughness',
        ]
        return {prefix + name: 0.0 for name in feature_names}

    # Basic statistics
    features[prefix + 'peak_flux'] = np.max(flux)
    peak_idx = np.argmax(flux)
    features[prefix + 'peak_time'] = time[peak_idx] - time[0]  # Relative to first observation
    features[prefix + 'mean_flux'] = np.mean(flux)
    features[prefix + 'std_flux'] = np.std(flux)
    features[prefix + 'median_flux'] = np.median(flux)

    # Shape descriptors
    centered = flux - np.mean(flux)
    std = np.std(flux) + 1e-8
    features[prefix + 'skewness'] = np.mean((centered / std) ** 3)
    features[prefix + 'kurtosis'] = np.mean((centered / std) ** 4) - 3

    # Rise time (time from first detection to peak)
    baseline = np.median(flux[:max(1, peak_idx // 4)]) if peak_idx > 3 else np.median(flux[:max(1, len(flux) // 4)])
    threshold = baseline + 2 * np.std(flux_err[:max(1, peak_idx // 4)] if peak_idx > 3 else flux_err)
    above_threshold = flux > threshold

    if np.any(above_threshold) and peak_idx > 0:
        first_detection_idx = np.where(above_threshold)[0][0]
        features[prefix + 'rise_time'] = time[peak_idx] - time[first_detection_idx]
    else:
        features[prefix + 'rise_time'] = 0.0

    # Decay rate (fit power law post-peak) - TDEs show t^(-5/3) ≈ -1.67
    if peak_idx < len(flux) - 3:
        post_peak_time = time[peak_idx + 1:] - time[peak_idx]
        post_peak_flux = flux[peak_idx + 1:]
        post_peak_err = flux_err[peak_idx + 1:]

        # Use enhanced power-law fitting
        alpha, chi2, alpha_dev = fit_power_law_decay(
            post_peak_time, post_peak_flux, post_peak_err
        )
        features[prefix + 'decay_rate'] = alpha
        features[prefix + 'decay_chi2'] = chi2 if chi2 != np.inf else 100.0  # Cap at 100
        features[prefix + 'decay_alpha_dev'] = alpha_dev if alpha_dev != np.inf else 10.0
    else:
        features[prefix + 'decay_rate'] = 0.0
        features[prefix + 'decay_chi2'] = 100.0
        features[prefix + 'decay_alpha_dev'] = 10.0

    # Amplitude ratio (peak / median)
    features[prefix + 'amplitude_ratio'] = features[prefix + 'peak_flux'] / (np.median(np.abs(flux)) + 1e-8)

    # Duration (time span of significant detections)
    features[prefix + 'duration'] = time[-1] - time[0]

    # Fraction of observations above baseline
    features[prefix + 'above_baseline_frac'] = np.mean(above_threshold)

    # Signal-to-noise ratio statistics
    snr = flux / (flux_err + 1e-8)
    features[prefix + 'snr_mean'] = np.mean(snr)
    features[prefix + 'snr_max'] = np.max(snr)

    # Smoothness features (TDEs are smooth, SNe have bumps)
    smoothness = compute_smoothness_features(flux, time)
    for key, value in smoothness.items():
        features[prefix + key] = value

    return features


def extract_cross_band_features(
    band_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    """
    Extract features that combine information across bands.

    Args:
        band_data: Dictionary mapping band name to (time, flux, flux_err) tuple

    Returns:
        Dictionary of feature name -> value
    """
    features = {}

    # Get peak times and fluxes for each band
    peak_times = {}
    peak_fluxes = {}

    for band, (time, flux, flux_err) in band_data.items():
        if len(flux) > 0:
            peak_idx = np.argmax(flux)
            peak_times[band] = time[peak_idx]
            peak_fluxes[band] = np.max(flux)

    # Color indices (approximate magnitudes as -2.5 * log10(flux))
    color_pairs = [('g', 'r'), ('r', 'i'), ('i', 'z'), ('u', 'g')]
    for b1, b2 in color_pairs:
        if b1 in peak_fluxes and b2 in peak_fluxes:
            f1, f2 = peak_fluxes[b1], peak_fluxes[b2]
            if f1 > 0 and f2 > 0:
                features[f'color_{b1}_{b2}'] = -2.5 * np.log10(f1 / f2)
            else:
                features[f'color_{b1}_{b2}'] = 0.0
        else:
            features[f'color_{b1}_{b2}'] = 0.0

    # Peak time differences (TDEs peak earlier in blue bands)
    if 'g' in peak_times and 'r' in peak_times:
        features['peak_time_g_minus_r'] = peak_times['g'] - peak_times['r']
    else:
        features['peak_time_g_minus_r'] = 0.0

    if 'u' in peak_times and 'z' in peak_times:
        features['peak_time_u_minus_z'] = peak_times['u'] - peak_times['z']
    else:
        features['peak_time_u_minus_z'] = 0.0

    # Total flux correlation across bands
    total_fluxes = [np.sum(np.abs(flux)) for _, (_, flux, _) in band_data.items() if len(flux) > 0]
    if len(total_fluxes) > 1:
        features['total_flux_std'] = np.std(total_fluxes)
        features['total_flux_mean'] = np.mean(total_fluxes)
    else:
        features['total_flux_std'] = 0.0
        features['total_flux_mean'] = 0.0

    # Color evolution features (how color changes over time)
    # TDEs typically evolve from blue to red
    color_evolution = _compute_color_evolution(band_data, peak_times)
    features.update(color_evolution)

    return features


def _interpolate_flux_at_time(time: np.ndarray, flux: np.ndarray, target_time: float) -> float:
    """Linearly interpolate flux at a given time."""
    if len(time) < 2 or target_time < time[0] or target_time > time[-1]:
        return np.nan
    return float(np.interp(target_time, time, flux))


def _compute_color_evolution(
    band_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    peak_times: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute color evolution features.

    TDEs typically start blue and evolve to redder colors as they fade.
    SNe have different color evolution patterns.
    """
    features = {}

    # Initialize defaults
    default_features = [
        'color_g_r_at_peak', 'color_g_r_at_30d', 'color_g_r_evolution_rate',
        'color_u_g_at_peak', 'color_u_g_at_30d', 'color_u_g_evolution_rate',
    ]
    for feat in default_features:
        features[feat] = 0.0

    # Get g and r band data for g-r color
    if 'g' in band_data and 'r' in band_data:
        g_time, g_flux, _ = band_data['g']
        r_time, r_flux, _ = band_data['r']

        if len(g_flux) > 2 and len(r_flux) > 2:
            # Use g-band peak as reference
            if 'g' in peak_times:
                peak_time = peak_times['g']

                # Color at peak
                g_at_peak = _interpolate_flux_at_time(g_time, g_flux, peak_time)
                r_at_peak = _interpolate_flux_at_time(r_time, r_flux, peak_time)

                if g_at_peak > 0 and r_at_peak > 0 and not np.isnan(g_at_peak) and not np.isnan(r_at_peak):
                    features['color_g_r_at_peak'] = -2.5 * np.log10(g_at_peak / r_at_peak)

                    # Color at +30 days
                    t_30d = peak_time + 30.0
                    g_at_30d = _interpolate_flux_at_time(g_time, g_flux, t_30d)
                    r_at_30d = _interpolate_flux_at_time(r_time, r_flux, t_30d)

                    if g_at_30d > 0 and r_at_30d > 0 and not np.isnan(g_at_30d) and not np.isnan(r_at_30d):
                        features['color_g_r_at_30d'] = -2.5 * np.log10(g_at_30d / r_at_30d)
                        # Evolution rate (mag per day)
                        features['color_g_r_evolution_rate'] = (
                            features['color_g_r_at_30d'] - features['color_g_r_at_peak']
                        ) / 30.0

    # Get u and g band data for u-g color (more sensitive to temperature)
    if 'u' in band_data and 'g' in band_data:
        u_time, u_flux, _ = band_data['u']
        g_time, g_flux, _ = band_data['g']

        if len(u_flux) > 2 and len(g_flux) > 2:
            if 'g' in peak_times:
                peak_time = peak_times['g']

                u_at_peak = _interpolate_flux_at_time(u_time, u_flux, peak_time)
                g_at_peak = _interpolate_flux_at_time(g_time, g_flux, peak_time)

                if u_at_peak > 0 and g_at_peak > 0 and not np.isnan(u_at_peak) and not np.isnan(g_at_peak):
                    features['color_u_g_at_peak'] = -2.5 * np.log10(u_at_peak / g_at_peak)

                    t_30d = peak_time + 30.0
                    u_at_30d = _interpolate_flux_at_time(u_time, u_flux, t_30d)
                    g_at_30d = _interpolate_flux_at_time(g_time, g_flux, t_30d)

                    if u_at_30d > 0 and g_at_30d > 0 and not np.isnan(u_at_30d) and not np.isnan(g_at_30d):
                        features['color_u_g_at_30d'] = -2.5 * np.log10(u_at_30d / g_at_30d)
                        features['color_u_g_evolution_rate'] = (
                            features['color_u_g_at_30d'] - features['color_u_g_at_peak']
                        ) / 30.0

    return features


def extract_all_features(
    object_id: str,
    log_row: pd.Series,
    lightcurve_df: pd.DataFrame,
    use_gp: bool = False,
    gp_interpolator: Optional['GPInterpolator'] = None,
    use_fats: bool = False,
    fats_extractor: Optional['FATSExtractor'] = None,
) -> Dict[str, float]:
    """
    Extract all features for a single object.

    Args:
        object_id: Object identifier
        log_row: Row from the log DataFrame with metadata
        lightcurve_df: DataFrame with light curve observations for this object
        use_gp: Whether to extract GP-based features
        gp_interpolator: GPInterpolator instance (required if use_gp=True)
        use_fats: Whether to extract FATS features
        fats_extractor: FATSExtractor instance (required if use_fats=True)

    Returns:
        Dictionary of feature name -> value
    """
    features = {'object_id': object_id}

    # Add metadata features (normalized)
    z = float(log_row['Z'])
    ebv = float(log_row['EBV'])
    features['Z_norm'] = (z - 0.6707) / 0.5393
    features['EBV_norm'] = (ebv - 0.0555) / 0.0613
    features['Z'] = z
    features['EBV'] = ebv

    # Extract per-band features
    band_data = {}
    for band in BANDS:
        band_lc = lightcurve_df[lightcurve_df['Filter'] == band].sort_values('Time (MJD)')

        if len(band_lc) == 0:
            time = np.array([])
            flux = np.array([])
            flux_err = np.array([])
        else:
            time = band_lc['Time (MJD)'].values.astype(np.float32)
            flux = band_lc['Flux'].values.astype(np.float32)
            flux_err = band_lc['Flux_err'].values.astype(np.float32)

            # Remove NaN values
            valid_mask = ~(np.isnan(time) | np.isnan(flux) | np.isnan(flux_err))
            time = time[valid_mask]
            flux = flux[valid_mask]
            flux_err = flux_err[valid_mask]

        band_data[band] = (time, flux, flux_err)

        # Extract per-band features
        band_features = extract_per_band_features(time, flux, flux_err, band)
        features.update(band_features)

    # Extract cross-band features
    cross_features = extract_cross_band_features(band_data)
    features.update(cross_features)

    # Extract GP-based features if requested
    if use_gp and gp_interpolator is not None:
        from .gp_interpolation import extract_gp_features
        gp_results = gp_interpolator.interpolate_object(lightcurve_df)
        gp_features = extract_gp_features(gp_results)
        features.update(gp_features)

    # Extract FATS features if requested
    if use_fats and fats_extractor is not None:
        fats_features = fats_extractor.extract_all_bands(lightcurve_df)
        features.update(fats_features)

    return features


def extract_features_batch(
    log_df: pd.DataFrame,
    lightcurves_df: pd.DataFrame,
    verbose: bool = True,
    use_gp: bool = False,
    use_fats: bool = False,
) -> pd.DataFrame:
    """
    Extract features for all objects in the dataset.

    Args:
        log_df: DataFrame with object metadata
        lightcurves_df: DataFrame with all light curve observations
        verbose: Whether to print progress
        use_gp: Whether to extract GP-based features (slower but better)
        use_fats: Whether to extract FATS features (slower, adds ~50 features/band)

    Returns:
        DataFrame with extracted features (one row per object)
    """
    all_features = []

    # Create GP interpolator if needed
    gp_interpolator = None
    if use_gp:
        from .gp_interpolation import GPInterpolator
        gp_interpolator = GPInterpolator(grid_spacing=1.0, n_restarts=2)
        if verbose:
            print("GP interpolation enabled - this will be slower but more accurate")

    # Create FATS extractor if needed
    fats_extractor = None
    if use_fats:
        from .fats_features import FATSExtractor
        fats_extractor = FATSExtractor()
        if verbose:
            print("FATS feature extraction enabled - this will be slower")

    # Group light curves by object_id for faster lookup
    lc_groups = lightcurves_df.groupby('object_id')

    for idx, (_, row) in enumerate(log_df.iterrows()):
        object_id = row['object_id']

        try:
            obj_lc = lc_groups.get_group(object_id)
        except KeyError:
            obj_lc = pd.DataFrame()

        features = extract_all_features(
            object_id, row, obj_lc,
            use_gp=use_gp,
            gp_interpolator=gp_interpolator,
            use_fats=use_fats,
            fats_extractor=fats_extractor,
        )

        # Add target if available
        if 'target' in row:
            features['target'] = row['target']

        all_features.append(features)

        if verbose and (idx + 1) % 100 == 0:
            print(f"Extracted features for {idx + 1}/{len(log_df)} objects")

    if verbose:
        print(f"Feature extraction complete: {len(all_features)} objects")

    return pd.DataFrame(all_features)


# List of feature column names (for consistent ordering)
def get_feature_columns(include_gp: bool = False, include_fats: bool = False) -> List[str]:
    """
    Get the list of feature column names (excluding object_id and target).

    Args:
        include_gp: Whether to include GP-based features
        include_fats: Whether to include FATS features

    Returns:
        List of feature column names
    """
    columns = ['Z', 'EBV', 'Z_norm', 'EBV_norm']

    # Per-band features (including new power-law and smoothness features)
    per_band_features = [
        'peak_flux', 'peak_time', 'mean_flux', 'std_flux', 'median_flux',
        'skewness', 'kurtosis', 'rise_time', 'decay_rate', 'amplitude_ratio',
        'duration', 'above_baseline_frac', 'snr_mean', 'snr_max',
        # New power-law features
        'decay_chi2', 'decay_alpha_dev',
        # Smoothness features
        'n_inflection_points', 'max_acceleration', 'monotonic_ratio', 'roughness',
    ]
    for band in BANDS:
        for feat in per_band_features:
            columns.append(f"{band}_{feat}")

    # Cross-band features
    columns.extend([
        'color_g_r', 'color_r_i', 'color_i_z', 'color_u_g',
        'peak_time_g_minus_r', 'peak_time_u_minus_z',
        'total_flux_std', 'total_flux_mean',
        # Color evolution features
        'color_g_r_at_peak', 'color_g_r_at_30d', 'color_g_r_evolution_rate',
        'color_u_g_at_peak', 'color_u_g_at_30d', 'color_u_g_evolution_rate',
    ])

    # GP-based features (from smooth interpolated curves)
    if include_gp:
        gp_features = [
            'gp_peak_flux', 'gp_peak_time', 'gp_peak_uncertainty',
            'gp_mean_uncertainty', 'gp_rise_time', 'gp_decay_time',
            'gp_fwhm', 'gp_asymmetry', 'gp_length_scale',
        ]
        for band in BANDS:
            for feat in gp_features:
                columns.append(f"{band}_{feat}")

    # FATS features (astronomical time-series features)
    if include_fats:
        from .fats_features import SELECTED_FATS_FEATURES
        for band in BANDS:
            for feat in SELECTED_FATS_FEATURES:
                columns.append(f"{band}_fats_{feat}")

    return columns
