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
                    'likelihood_ratio_drw_vs_gp']

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

    return features


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

        all_features.append(obj_features)

        if verbose and (idx + 1) % 100 == 0:
            print(f"Extracted GP/DRW features for {idx + 1}/{n_objects} objects")

    return pd.DataFrame(all_features)


def get_gp_drw_feature_columns(bands: list = ['g', 'r', 'i']) -> list:
    """Return list of GP/DRW feature column names."""
    base_features = [
        'drw_tau', 'drw_sigma', 'drw_SF_inf', 'drw_log_likelihood',
        'gp_matern_length_scale', 'gp_matern_amplitude', 'gp_log_likelihood',
        'likelihood_ratio_drw_vs_gp'
    ]

    columns = []
    for band in bands:
        for feat in base_features:
            columns.append(f'{band}_{feat}')

    # Combined features
    for feat in base_features:
        columns.append(f'combined_{feat}')

    return columns
