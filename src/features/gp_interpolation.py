"""
Gaussian Process interpolation for irregular light curves.

Fits a GP to each band's light curve and interpolates to a regular time grid.
This provides:
1. Smooth, evenly-sampled data for neural networks
2. Uncertainty estimates at each interpolated point
3. Better feature extraction from continuous curves

Usage:
    from src.features.gp_interpolation import GPInterpolator

    interpolator = GPInterpolator(grid_spacing=1.0)
    result = interpolator.interpolate_object(object_lc_df)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, Matern, ConstantKernel as C
)

# LSST photometric bands
BANDS = ['u', 'g', 'r', 'i', 'z', 'y']


@dataclass
class GPResult:
    """Result of GP interpolation for a single band."""
    time_grid: np.ndarray       # Regular time grid
    flux_mean: np.ndarray       # Interpolated flux (GP mean)
    flux_std: np.ndarray        # Uncertainty (GP std)
    n_observations: int         # Original number of observations
    time_span: float            # Total time coverage
    kernel_params: dict         # Fitted kernel parameters
    success: bool               # Whether fitting succeeded


class GPInterpolator:
    """
    Gaussian Process interpolator for astronomical light curves.

    Uses a Matern kernel (good for physical processes) plus white noise
    to model the light curve and interpolate to a regular grid.
    """

    def __init__(
        self,
        grid_spacing: float = 1.0,
        min_observations: int = 5,
        max_grid_points: int = 500,
        kernel_type: str = 'matern',
        normalize: bool = True,
        n_restarts: int = 3,
    ):
        """
        Initialize the GP interpolator.

        Args:
            grid_spacing: Spacing between interpolated points (days)
            min_observations: Minimum observations required for GP fit
            max_grid_points: Maximum number of grid points to generate
            kernel_type: 'matern' or 'rbf'
            normalize: Whether to normalize flux before fitting
            n_restarts: Number of optimizer restarts for kernel fitting
        """
        self.grid_spacing = grid_spacing
        self.min_observations = min_observations
        self.max_grid_points = max_grid_points
        self.kernel_type = kernel_type
        self.normalize = normalize
        self.n_restarts = n_restarts

    def _create_kernel(self, time_scale: float, flux_scale: float) -> object:
        """Create the GP kernel based on data characteristics."""
        # Length scale bounds based on typical transient timescales
        length_scale_bounds = (1.0, 200.0)  # 1 day to 200 days

        if self.kernel_type == 'matern':
            # Matern 3/2 is good for smooth but not infinitely differentiable functions
            kernel = (
                C(flux_scale**2, (1e-4, 1e4)) *
                Matern(length_scale=time_scale, length_scale_bounds=length_scale_bounds, nu=1.5) +
                WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
            )
        else:
            # RBF (squared exponential) - very smooth
            kernel = (
                C(flux_scale**2, (1e-4, 1e4)) *
                RBF(length_scale=time_scale, length_scale_bounds=length_scale_bounds) +
                WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
            )

        return kernel

    def _fit_single_band(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
    ) -> GPResult:
        """
        Fit GP to a single band and interpolate.

        Args:
            time: Observation times (MJD)
            flux: Flux values
            flux_err: Flux uncertainties

        Returns:
            GPResult with interpolated data
        """
        # Check minimum observations
        if len(time) < self.min_observations:
            return GPResult(
                time_grid=np.array([]),
                flux_mean=np.array([]),
                flux_std=np.array([]),
                n_observations=len(time),
                time_span=0.0,
                kernel_params={},
                success=False,
            )

        # Sort by time
        sort_idx = np.argsort(time)
        time = time[sort_idx]
        flux = flux[sort_idx]
        flux_err = flux_err[sort_idx]

        # Normalize time to start from 0
        time_offset = time[0]
        time_norm = time - time_offset
        time_span = time_norm[-1] - time_norm[0]

        # Normalize flux if requested
        if self.normalize:
            flux_mean_orig = np.mean(flux)
            flux_std_orig = np.std(flux)
            if flux_std_orig < 1e-10:
                flux_std_orig = 1.0
            flux_norm = (flux - flux_mean_orig) / flux_std_orig
            flux_err_norm = flux_err / flux_std_orig
        else:
            flux_mean_orig = 0.0
            flux_std_orig = 1.0
            flux_norm = flux
            flux_err_norm = flux_err

        # Create regular time grid
        n_grid = min(int(time_span / self.grid_spacing) + 1, self.max_grid_points)
        if n_grid < 2:
            n_grid = 2
        time_grid = np.linspace(0, time_span, n_grid)

        # Estimate initial kernel parameters
        time_scale = max(10.0, time_span / 10)  # Initial length scale
        flux_scale = np.std(flux_norm) if np.std(flux_norm) > 0 else 1.0

        # Create kernel and GP
        kernel = self._create_kernel(time_scale, flux_scale)

        # Use flux errors as alpha (noise variance per point)
        alpha = np.clip(flux_err_norm**2, 1e-10, 1e2)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=False,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(time_norm.reshape(-1, 1), flux_norm)

            # Predict on regular grid
            flux_pred, flux_pred_std = gp.predict(
                time_grid.reshape(-1, 1),
                return_std=True
            )

            # Denormalize
            flux_pred = flux_pred * flux_std_orig + flux_mean_orig
            flux_pred_std = flux_pred_std * flux_std_orig

            # Shift time grid back to original offset
            time_grid_abs = time_grid + time_offset

            # Extract kernel parameters
            kernel_params = {
                'length_scale': float(gp.kernel_.get_params().get('k1__k2__length_scale', 0)),
                'noise_level': float(gp.kernel_.get_params().get('k2__noise_level', 0)),
            }

            return GPResult(
                time_grid=time_grid_abs,
                flux_mean=flux_pred,
                flux_std=flux_pred_std,
                n_observations=len(time),
                time_span=time_span,
                kernel_params=kernel_params,
                success=True,
            )

        except Exception as e:
            # GP fitting failed - return empty result
            return GPResult(
                time_grid=np.array([]),
                flux_mean=np.array([]),
                flux_std=np.array([]),
                n_observations=len(time),
                time_span=time_span,
                kernel_params={'error': str(e)},
                success=False,
            )

    def interpolate_object(
        self,
        lightcurve_df: pd.DataFrame,
    ) -> Dict[str, GPResult]:
        """
        Interpolate all bands for a single object.

        Args:
            lightcurve_df: DataFrame with columns ['Time (MJD)', 'Flux', 'Flux_err', 'Filter']

        Returns:
            Dictionary mapping band name to GPResult
        """
        results = {}

        for band in BANDS:
            band_data = lightcurve_df[lightcurve_df['Filter'] == band].copy()

            if len(band_data) == 0:
                results[band] = GPResult(
                    time_grid=np.array([]),
                    flux_mean=np.array([]),
                    flux_std=np.array([]),
                    n_observations=0,
                    time_span=0.0,
                    kernel_params={},
                    success=False,
                )
                continue

            time = band_data['Time (MJD)'].values.astype(np.float64)
            flux = band_data['Flux'].values.astype(np.float64)
            flux_err = band_data['Flux_err'].values.astype(np.float64)

            # Remove NaN values
            valid = ~(np.isnan(time) | np.isnan(flux) | np.isnan(flux_err))
            time = time[valid]
            flux = flux[valid]
            flux_err = flux_err[valid]

            # Ensure positive flux errors
            flux_err = np.clip(flux_err, 1e-10, None)

            results[band] = self._fit_single_band(time, flux, flux_err)

        return results

    def interpolate_to_common_grid(
        self,
        lightcurve_df: pd.DataFrame,
        grid_points: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate all bands to a common normalized time grid.

        Useful for neural network input where we want fixed-size arrays.

        Args:
            lightcurve_df: DataFrame with light curve data
            grid_points: Number of points in the output grid

        Returns:
            Tuple of:
            - time_grid: (grid_points,) normalized time grid [0, 1]
            - flux_array: (n_bands, grid_points) interpolated fluxes
            - std_array: (n_bands, grid_points) uncertainties
        """
        # First, find global time range across all bands
        all_times = lightcurve_df['Time (MJD)'].dropna().values
        if len(all_times) == 0:
            return (
                np.linspace(0, 1, grid_points),
                np.zeros((len(BANDS), grid_points)),
                np.ones((len(BANDS), grid_points)),
            )

        t_min, t_max = all_times.min(), all_times.max()
        time_span = t_max - t_min
        if time_span < 1e-6:
            time_span = 1.0

        # Create common time grid
        time_grid_abs = np.linspace(t_min, t_max, grid_points)
        time_grid_norm = np.linspace(0, 1, grid_points)

        flux_array = np.zeros((len(BANDS), grid_points))
        std_array = np.ones((len(BANDS), grid_points))

        for i, band in enumerate(BANDS):
            band_data = lightcurve_df[lightcurve_df['Filter'] == band].copy()

            if len(band_data) < self.min_observations:
                continue

            time = band_data['Time (MJD)'].values.astype(np.float64)
            flux = band_data['Flux'].values.astype(np.float64)
            flux_err = band_data['Flux_err'].values.astype(np.float64)

            # Remove NaN
            valid = ~(np.isnan(time) | np.isnan(flux) | np.isnan(flux_err))
            if valid.sum() < self.min_observations:
                continue

            time = time[valid]
            flux = flux[valid]
            flux_err = np.clip(flux_err[valid], 1e-10, None)

            # Fit GP
            try:
                # Normalize
                flux_mean = np.mean(flux)
                flux_std = np.std(flux)
                if flux_std < 1e-10:
                    flux_std = 1.0
                flux_norm = (flux - flux_mean) / flux_std
                flux_err_norm = flux_err / flux_std

                kernel = self._create_kernel(time_span / 10, 1.0)
                alpha = np.clip(flux_err_norm**2, 1e-10, 1e2)

                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    n_restarts_optimizer=self.n_restarts,
                    normalize_y=False,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp.fit(time.reshape(-1, 1), flux_norm)

                pred_mean, pred_std = gp.predict(
                    time_grid_abs.reshape(-1, 1),
                    return_std=True
                )

                # Denormalize
                flux_array[i] = pred_mean * flux_std + flux_mean
                std_array[i] = pred_std * flux_std

            except Exception:
                # If GP fails, use linear interpolation as fallback
                flux_array[i] = np.interp(time_grid_abs, time, flux)
                std_array[i] = np.ones(grid_points) * np.mean(flux_err)

        return time_grid_norm, flux_array, std_array


def extract_gp_features(gp_results: Dict[str, GPResult]) -> Dict[str, float]:
    """
    Extract features from GP-interpolated light curves.

    These features leverage the smooth, uncertainty-aware GP predictions.

    Args:
        gp_results: Dictionary of GPResult per band

    Returns:
        Dictionary of feature name -> value
    """
    features = {}

    for band, result in gp_results.items():
        prefix = f"{band}_gp_"

        if not result.success or len(result.flux_mean) < 3:
            # Fill with defaults
            features[prefix + 'peak_flux'] = 0.0
            features[prefix + 'peak_time'] = 0.0
            features[prefix + 'peak_uncertainty'] = 1.0
            features[prefix + 'mean_uncertainty'] = 1.0
            features[prefix + 'rise_time'] = 0.0
            features[prefix + 'decay_time'] = 0.0
            features[prefix + 'fwhm'] = 0.0
            features[prefix + 'asymmetry'] = 0.0
            features[prefix + 'length_scale'] = 0.0
            continue

        flux = result.flux_mean
        flux_std = result.flux_std
        time = result.time_grid

        # Peak properties (from smooth GP curve)
        peak_idx = np.argmax(flux)
        features[prefix + 'peak_flux'] = float(flux[peak_idx])
        features[prefix + 'peak_time'] = float(time[peak_idx] - time[0])
        features[prefix + 'peak_uncertainty'] = float(flux_std[peak_idx])

        # Mean uncertainty (measure of data quality)
        features[prefix + 'mean_uncertainty'] = float(np.mean(flux_std))

        # Rise and decay times (from half-max)
        half_max = (flux[peak_idx] + np.min(flux)) / 2
        above_half = flux > half_max

        if np.any(above_half[:peak_idx]) and peak_idx > 0:
            first_above = np.where(above_half[:peak_idx])[0][0]
            features[prefix + 'rise_time'] = float(time[peak_idx] - time[first_above])
        else:
            features[prefix + 'rise_time'] = 0.0

        if np.any(above_half[peak_idx:]) and peak_idx < len(flux) - 1:
            post_peak_above = np.where(above_half[peak_idx:])[0]
            if len(post_peak_above) > 0:
                last_above = peak_idx + post_peak_above[-1]
                features[prefix + 'decay_time'] = float(time[last_above] - time[peak_idx])
            else:
                features[prefix + 'decay_time'] = 0.0
        else:
            features[prefix + 'decay_time'] = 0.0

        # FWHM
        features[prefix + 'fwhm'] = features[prefix + 'rise_time'] + features[prefix + 'decay_time']

        # Asymmetry (rise vs decay)
        if features[prefix + 'fwhm'] > 0:
            features[prefix + 'asymmetry'] = (
                (features[prefix + 'decay_time'] - features[prefix + 'rise_time']) /
                features[prefix + 'fwhm']
            )
        else:
            features[prefix + 'asymmetry'] = 0.0

        # GP kernel length scale (characteristic timescale)
        features[prefix + 'length_scale'] = float(
            result.kernel_params.get('length_scale', 0.0)
        )

    return features
