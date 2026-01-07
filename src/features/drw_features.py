"""
Damped Random Walk (DRW) and GP model fitting for light curves.

DRW is the standard model for AGN variability:
    dX = -X/tau * dt + sigma * sqrt(dt) * dW

GP with Matern kernel is used for transient modeling in MALLORN.

Features extracted:
- DRW parameters: tau (timescale), sigma (amplitude), SF_inf
- GP Matern parameters: length_scale, amplitude
- Model comparison: DRW log-likelihood, GP log-likelihood, likelihood ratio

Reference: MALLORN paper (arXiv:2512.04946)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cho_solve, cho_factor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
import warnings
from typing import Dict, Tuple, Optional
from tqdm import tqdm


class DRWFitter:
    """
    Fit Damped Random Walk model to light curve.

    DRW covariance: K(t1, t2) = SF_inf^2 * exp(-|t1-t2|/tau)
    where SF_inf = sigma * sqrt(tau / 2)
    """

    def __init__(self, tau_bounds: Tuple[float, float] = (1, 500),
                 sigma_bounds: Tuple[float, float] = (0.01, 10)):
        self.tau_bounds = tau_bounds
        self.sigma_bounds = sigma_bounds
        self.tau_ = None
        self.sigma_ = None
        self.log_likelihood_ = None
        self.fit_success_ = False

    def _drw_covariance(self, t: np.ndarray, tau: float, sigma: float) -> np.ndarray:
        """Compute DRW covariance matrix."""
        dt = np.abs(t[:, None] - t[None, :])
        SF_inf = sigma * np.sqrt(tau / 2)
        return SF_inf**2 * np.exp(-dt / tau)

    def _neg_log_likelihood(self, params: np.ndarray, t: np.ndarray,
                            y: np.ndarray, y_err: np.ndarray) -> float:
        """Negative log-likelihood for optimization."""
        tau, sigma = params

        if tau <= 0 or sigma <= 0:
            return 1e10

        try:
            K = self._drw_covariance(t, tau, sigma)
            K += np.diag(y_err**2)  # Add measurement noise

            L, lower = cho_factor(K, lower=True)
            alpha = cho_solve((L, lower), y)

            log_det = 2 * np.sum(np.log(np.diag(L)))
            nll = 0.5 * (y @ alpha + log_det + len(y) * np.log(2 * np.pi))

            return nll
        except Exception:
            return 1e10

    def fit(self, t: np.ndarray, y: np.ndarray, y_err: np.ndarray) -> 'DRWFitter':
        """Fit DRW model to light curve."""
        if len(t) < 5:
            self.tau_ = np.nan
            self.sigma_ = np.nan
            self.log_likelihood_ = np.nan
            return self

        # Normalize time and flux
        t_norm = t - t.min()
        y_mean = np.mean(y)
        y_norm = y - y_mean

        # Initial guess based on data
        duration = t_norm.max()
        std_y = np.std(y_norm)
        x0 = [min(duration / 4, 100), std_y]

        # Optimize
        bounds = [self.tau_bounds, self.sigma_bounds]

        try:
            result = minimize(
                self._neg_log_likelihood,
                x0,
                args=(t_norm, y_norm, y_err),
                method='L-BFGS-B',
                bounds=bounds,
            )

            if result.success:
                self.tau_, self.sigma_ = result.x
                self.log_likelihood_ = -result.fun
                self.fit_success_ = True
            else:
                self.tau_, self.sigma_ = np.nan, np.nan
                self.log_likelihood_ = np.nan
        except Exception:
            self.tau_, self.sigma_ = np.nan, np.nan
            self.log_likelihood_ = np.nan

        return self

    def get_features(self) -> Dict[str, float]:
        """Return DRW features."""
        SF_inf = self.sigma_ * np.sqrt(self.tau_ / 2) if np.isfinite(self.tau_) else np.nan
        return {
            'drw_tau': self.tau_,
            'drw_sigma': self.sigma_,
            'drw_SF_inf': SF_inf,
            'drw_log_likelihood': self.log_likelihood_,
        }


class GPTransientFitter:
    """
    Fit GP with Matern kernel (as used in MALLORN generation for transients).
    """

    def __init__(self, nu: float = 1.5):
        self.nu = nu
        self.length_scale_ = None
        self.amplitude_ = None
        self.log_likelihood_ = None
        self.fit_success_ = False

    def fit(self, t: np.ndarray, y: np.ndarray, y_err: np.ndarray) -> 'GPTransientFitter':
        """Fit GP with Matern kernel."""
        if len(t) < 5:
            self.length_scale_ = np.nan
            self.amplitude_ = np.nan
            self.log_likelihood_ = np.nan
            return self

        t_norm = (t - t.min()).reshape(-1, 1)
        y_mean = np.mean(y)
        y_norm = y - y_mean
        y_std = np.std(y_norm) or 1.0

        # Define kernel: amplitude * Matern + white noise
        kernel = (
            C(y_std**2, (1e-4, 1e4)) *
            Matern(length_scale=30, length_scale_bounds=(1, 300), nu=self.nu) +
            WhiteKernel(noise_level=np.mean(y_err)**2, noise_level_bounds=(1e-8, 1))
        )

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=y_err**2,
            n_restarts_optimizer=2,
            normalize_y=False,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(t_norm, y_norm)

            self.log_likelihood_ = gp.log_marginal_likelihood_value_

            # Extract kernel parameters
            kernel_params = gp.kernel_.get_params()
            self.length_scale_ = kernel_params.get('k1__k2__length_scale', np.nan)
            self.amplitude_ = np.sqrt(kernel_params.get('k1__k1__constant_value', np.nan))
            self.fit_success_ = True

        except Exception:
            self.length_scale_ = np.nan
            self.amplitude_ = np.nan
            self.log_likelihood_ = np.nan

        return self

    def get_features(self) -> Dict[str, float]:
        """Return GP features."""
        return {
            'gp_matern_length_scale': self.length_scale_,
            'gp_matern_amplitude': self.amplitude_,
            'gp_log_likelihood': self.log_likelihood_,
        }


def extract_gp_drw_features_single(time: np.ndarray, flux: np.ndarray,
                                    flux_err: np.ndarray, band: str = '') -> Dict[str, float]:
    """
    Extract both GP and DRW features for a single light curve.

    Returns features including likelihood ratio for model comparison.
    """
    prefix = f"{band}_" if band else ''
    features = {}

    # Default values for insufficient data
    default_keys = ['drw_tau', 'drw_sigma', 'drw_SF_inf', 'drw_log_likelihood',
                    'gp_matern_length_scale', 'gp_matern_amplitude', 'gp_log_likelihood',
                    'likelihood_ratio_drw_vs_gp',
                    # Normalized likelihood features
                    'drw_ll_per_point', 'gp_ll_per_point', 'lr_per_point',
                    'delta_aic_drw_minus_gp']

    if len(time) < 10:
        for key in default_keys:
            features[prefix + key] = 0.0
        return features

    # Clean data
    valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    time, flux, flux_err = time[valid], flux[valid], flux_err[valid]

    if len(time) < 10:
        for key in default_keys:
            features[prefix + key] = 0.0
        return features

    # Fit DRW
    drw = DRWFitter()
    drw.fit(time, flux, flux_err)
    drw_features = drw.get_features()

    # Fit GP Matern
    gp = GPTransientFitter(nu=1.5)
    gp.fit(time, flux, flux_err)
    gp_features = gp.get_features()

    # Combine features with prefix
    for key, val in drw_features.items():
        features[prefix + key] = val if np.isfinite(val) else 0.0
    for key, val in gp_features.items():
        features[prefix + key] = val if np.isfinite(val) else 0.0

    # Likelihood ratio (positive = more DRW-like/AGN-like)
    if np.isfinite(drw.log_likelihood_) and np.isfinite(gp.log_likelihood_):
        features[prefix + 'likelihood_ratio_drw_vs_gp'] = drw.log_likelihood_ - gp.log_likelihood_
    else:
        features[prefix + 'likelihood_ratio_drw_vs_gp'] = 0.0

    # Normalized likelihood features (per-point, stabilizes across different LC lengths)
    n_obs = len(time)  # Number of observations used in fitting
    k_drw = 2  # DRW parameters: tau, sigma
    k_gp = 3   # GP parameters: length_scale, amplitude, noise

    if n_obs > 0 and np.isfinite(drw.log_likelihood_):
        features[prefix + 'drw_ll_per_point'] = drw.log_likelihood_ / n_obs
    else:
        features[prefix + 'drw_ll_per_point'] = 0.0

    if n_obs > 0 and np.isfinite(gp.log_likelihood_):
        features[prefix + 'gp_ll_per_point'] = gp.log_likelihood_ / n_obs
    else:
        features[prefix + 'gp_ll_per_point'] = 0.0

    if n_obs > 0 and np.isfinite(drw.log_likelihood_) and np.isfinite(gp.log_likelihood_):
        features[prefix + 'lr_per_point'] = (drw.log_likelihood_ - gp.log_likelihood_) / n_obs
    else:
        features[prefix + 'lr_per_point'] = 0.0

    # AIC-style comparison (negative delta_aic means DRW preferred)
    if np.isfinite(drw.log_likelihood_) and np.isfinite(gp.log_likelihood_):
        aic_drw = 2 * k_drw - 2 * drw.log_likelihood_
        aic_gp = 2 * k_gp - 2 * gp.log_likelihood_
        features[prefix + 'delta_aic_drw_minus_gp'] = aic_drw - aic_gp
    else:
        features[prefix + 'delta_aic_drw_minus_gp'] = 0.0

    return features


def compute_gp_drw_cross_band_features(obj_features: dict, bands: list = ['g', 'r', 'i']) -> dict:
    """
    Compute cross-band consistency features from per-band GP/DRW fits.

    These features encode "achromatic DRW" vs "chromatic transient" behavior.
    AGN should have consistent parameters across bands.

    Args:
        obj_features: Dictionary containing per-band features (e.g., 'g_drw_tau', etc.)
        bands: List of bands to aggregate

    Returns:
        Dictionary of cross-band features
    """
    cross_features = {}

    # Helper to get per-band values, converting 0.0 (our default for missing) to NaN
    def get_band_values(feature_name):
        values = []
        for band in bands:
            key = f'{band}_{feature_name}'
            val = obj_features.get(key, 0.0)
            # Treat 0.0 as missing for parameters that should never be exactly 0
            if val == 0.0 and feature_name in ['drw_tau', 'drw_sigma', 'drw_SF_inf',
                                                 'gp_matern_length_scale', 'gp_matern_amplitude']:
                val = np.nan
            values.append(val)
        return np.array(values)

    # Timescale consistency (AGN have consistent tau across bands)
    tau_values = get_band_values('drw_tau')
    tau_mean = np.nanmean(tau_values)
    tau_std = np.nanstd(tau_values)
    cross_features['tau_mean_across_bands'] = tau_mean if np.isfinite(tau_mean) else 0.0
    cross_features['tau_std_across_bands'] = tau_std if np.isfinite(tau_std) else 0.0
    cross_features['tau_cv_across_bands'] = tau_std / (tau_mean + 1e-8) if np.isfinite(tau_mean) and np.isfinite(tau_std) else 0.0

    # Amplitude consistency
    sigma_values = get_band_values('drw_sigma')
    sigma_mean = np.nanmean(sigma_values)
    sigma_std = np.nanstd(sigma_values)
    cross_features['sigma_cv_across_bands'] = sigma_std / (sigma_mean + 1e-8) if np.isfinite(sigma_mean) and np.isfinite(sigma_std) else 0.0

    # Structure function infinity consistency
    sf_values = get_band_values('drw_SF_inf')
    sf_mean = np.nanmean(sf_values)
    sf_std = np.nanstd(sf_values)
    cross_features['sf_inf_cv_across_bands'] = sf_std / (sf_mean + 1e-8) if np.isfinite(sf_mean) and np.isfinite(sf_std) else 0.0

    # Likelihood ratio consistency (model preference should agree across bands)
    lr_values = np.array([obj_features.get(f'{band}_likelihood_ratio_drw_vs_gp', 0.0) for band in bands])
    cross_features['lr_mean'] = np.nanmean(lr_values)
    cross_features['lr_std'] = np.nanstd(lr_values)
    cross_features['lr_max'] = np.nanmax(lr_values)
    # Fraction of bands where DRW is preferred (LR > 0)
    cross_features['lr_sign_agreement'] = np.sum(lr_values > 0) / len(bands)

    # Combined vs per-band disagreement features
    combined_tau = obj_features.get('combined_drw_tau', 0.0)
    if combined_tau == 0.0:
        combined_tau = np.nan
    cross_features['delta_tau_combined_minus_mean'] = (combined_tau - tau_mean) if np.isfinite(combined_tau) and np.isfinite(tau_mean) else 0.0

    combined_lr = obj_features.get('combined_likelihood_ratio_drw_vs_gp', 0.0)
    lr_mean_val = cross_features['lr_mean']
    cross_features['abs_delta_lr_combined_minus_lr_mean'] = abs(combined_lr - lr_mean_val) if np.isfinite(lr_mean_val) else 0.0
    cross_features['combined_prefers_drw'] = 1.0 if combined_lr > 0 else 0.0

    return cross_features


def extract_gp_drw_features_batch(log_df: pd.DataFrame, lc_df: pd.DataFrame,
                                   bands: list = ['g', 'r', 'i'],
                                   verbose: bool = True) -> pd.DataFrame:
    """
    Extract GP/DRW features for all objects.

    Args:
        log_df: Object metadata
        lc_df: Light curve data
        bands: Bands to fit (g, r, i are most useful in MALLORN)

    Returns:
        DataFrame with GP/DRW features
    """
    all_features = []
    lc_groups = lc_df.groupby('object_id')
    n_objects = len(log_df)

    for idx, (_, row) in enumerate(log_df.iterrows()):
        object_id = row['object_id']
        obj_features = {'object_id': object_id}

        try:
            obj_lc = lc_groups.get_group(object_id)
        except KeyError:
            obj_lc = pd.DataFrame()

        for band in bands:
            band_lc = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

            if len(band_lc) >= 10:
                t = band_lc['Time (MJD)'].values
                f = band_lc['Flux'].values
                f_err = band_lc['Flux_err'].values

                features = extract_gp_drw_features_single(t, f, f_err, band)
            else:
                features = extract_gp_drw_features_single(
                    np.array([]), np.array([]), np.array([]), band
                )

            obj_features.update(features)

        # Also fit combined light curve (all bands concatenated)
        if len(obj_lc) >= 10:
            t_all = obj_lc['Time (MJD)'].values
            f_all = obj_lc['Flux'].values
            f_err_all = obj_lc['Flux_err'].values
            combined_features = extract_gp_drw_features_single(
                t_all, f_all, f_err_all, 'combined'
            )
        else:
            combined_features = extract_gp_drw_features_single(
                np.array([]), np.array([]), np.array([]), 'combined'
            )
        obj_features.update(combined_features)

        # Compute cross-band consistency features
        cross_band_features = compute_gp_drw_cross_band_features(obj_features, bands=bands)
        obj_features.update(cross_band_features)

        # Add rest-frame time features using redshift
        z = row.get('Z', 0.0)
        if z is None or not np.isfinite(z):
            z = 0.0
        rest_factor = 1 + z

        # Global redshift feature
        obj_features['log1p_z'] = np.log1p(z)

        # Rest-frame timescale features for each band
        for band in bands:
            tau_obs = obj_features.get(f'{band}_drw_tau', 0.0)
            length_scale_obs = obj_features.get(f'{band}_gp_matern_length_scale', 0.0)

            obj_features[f'{band}_tau_rest'] = tau_obs / rest_factor if tau_obs > 0 else 0.0
            obj_features[f'{band}_gp_length_scale_rest'] = length_scale_obs / rest_factor if length_scale_obs > 0 else 0.0

        # Rest-frame for combined fit
        combined_tau = obj_features.get('combined_drw_tau', 0.0)
        combined_ls = obj_features.get('combined_gp_matern_length_scale', 0.0)
        obj_features['combined_tau_rest'] = combined_tau / rest_factor if combined_tau > 0 else 0.0
        obj_features['combined_gp_length_scale_rest'] = combined_ls / rest_factor if combined_ls > 0 else 0.0

        all_features.append(obj_features)

        if verbose and (idx + 1) % 100 == 0:
            print(f"Extracted GP/DRW features for {idx + 1}/{n_objects} objects")

    return pd.DataFrame(all_features)


def get_gp_drw_feature_columns(bands: list = ['g', 'r', 'i']) -> list:
    """Return list of GP/DRW feature column names."""
    base_features = [
        'drw_tau', 'drw_sigma', 'drw_SF_inf', 'drw_log_likelihood',
        'gp_matern_length_scale', 'gp_matern_amplitude', 'gp_log_likelihood',
        'likelihood_ratio_drw_vs_gp',
        # Normalized likelihood features
        'drw_ll_per_point', 'gp_ll_per_point', 'lr_per_point',
        'delta_aic_drw_minus_gp',
    ]

    columns = []
    for band in bands:
        for feat in base_features:
            columns.append(f'{band}_{feat}')

    # Combined features
    for feat in base_features:
        columns.append(f'combined_{feat}')

    # Cross-band consistency features
    cross_band_features = [
        'tau_mean_across_bands', 'tau_std_across_bands', 'tau_cv_across_bands',
        'sigma_cv_across_bands', 'sf_inf_cv_across_bands',
        'lr_mean', 'lr_std', 'lr_max', 'lr_sign_agreement',
        'delta_tau_combined_minus_mean', 'abs_delta_lr_combined_minus_lr_mean',
        'combined_prefers_drw',
    ]
    columns.extend(cross_band_features)

    # Rest-frame time features (per-band)
    for band in bands:
        columns.append(f'{band}_tau_rest')
        columns.append(f'{band}_gp_length_scale_rest')

    # Combined rest-frame features
    columns.append('combined_tau_rest')
    columns.append('combined_gp_length_scale_rest')

    # Global redshift feature
    columns.append('log1p_z')

    return columns
