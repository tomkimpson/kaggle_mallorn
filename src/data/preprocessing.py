"""
Preprocessing utilities for MALLORN light curves.

Includes de-extinction correction using Fitzpatrick99 law.
"""

import numpy as np
from typing import Dict, Optional

try:
    from extinction import fitzpatrick99
    EXTINCTION_AVAILABLE = True
except ImportError:
    EXTINCTION_AVAILABLE = False


# LSST filter effective wavelengths in Angstroms (from SVO Filter Profile Service)
BAND_WAVELENGTHS = {
    'u': 3641.0,
    'g': 4704.0,
    'r': 6155.0,
    'i': 7504.0,
    'z': 8695.0,
    'y': 10056.0
}

# Standard Milky Way R_V value
R_V = 3.1


def apply_extinction_correction(
    flux: np.ndarray,
    flux_err: np.ndarray,
    ebv: float,
    band: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply extinction correction to flux values using Fitzpatrick99 law.

    The raw flux values are affected by dust extinction in the Milky Way.
    This function corrects for that extinction using the E(B-V) value.

    Args:
        flux: Array of flux values in microjansky
        flux_err: Array of flux error values
        ebv: E(B-V) extinction coefficient
        band: Filter band ('u', 'g', 'r', 'i', 'z', 'y')

    Returns:
        Tuple of (corrected_flux, corrected_flux_err)
    """
    if not EXTINCTION_AVAILABLE:
        raise ImportError(
            "extinction package not installed. "
            "Install with: pip install extinction"
        )

    if band not in BAND_WAVELENGTHS:
        raise ValueError(f"Unknown band: {band}. Must be one of {list(BAND_WAVELENGTHS.keys())}")

    # Get wavelength for this band
    wavelength = np.array([BAND_WAVELENGTHS[band]])

    # Calculate extinction in magnitudes at this wavelength
    a_lambda = fitzpatrick99(wavelength, ebv * R_V)[0]

    # Convert extinction from magnitudes to flux factor
    # m = -2.5 * log10(F) -> F_corrected = F * 10^(A_lambda / 2.5)
    correction_factor = 10 ** (a_lambda / 2.5)

    corrected_flux = flux * correction_factor
    corrected_flux_err = flux_err * correction_factor

    return corrected_flux, corrected_flux_err


def apply_extinction_correction_all_bands(
    object_data: Dict,
    ebv: float
) -> Dict:
    """
    Apply extinction correction to all bands for an object.

    Args:
        object_data: Dictionary with band data (from load_object_lightcurve)
        ebv: E(B-V) extinction coefficient

    Returns:
        Modified object_data with corrected flux values
    """
    from .loader import BANDS

    result = {'metadata': object_data['metadata'].copy()}

    for band in BANDS:
        if band not in object_data:
            continue

        band_data = object_data[band]
        corrected_flux, corrected_flux_err = apply_extinction_correction(
            band_data['flux'],
            band_data['flux_err'],
            ebv,
            band
        )

        result[band] = {
            'time': band_data['time'].copy(),
            'flux': corrected_flux,
            'flux_err': corrected_flux_err,
        }

    return result


def normalize_time(time: np.ndarray, reference: str = 'first') -> np.ndarray:
    """
    Normalize time values to start from a reference point.

    Args:
        time: Array of MJD time values
        reference: 'first' to subtract first observation, 'min' for minimum

    Returns:
        Normalized time array
    """
    if len(time) == 0:
        return time

    if reference == 'first':
        return time - time[0]
    elif reference == 'min':
        return time - time.min()
    else:
        raise ValueError(f"Unknown reference: {reference}")


def normalize_flux(
    flux: np.ndarray,
    flux_err: np.ndarray,
    method: str = 'max'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize flux values.

    Args:
        flux: Array of flux values
        flux_err: Array of flux error values
        method: Normalization method
            - 'max': Divide by max absolute value
            - 'std': Divide by standard deviation
            - 'peak': Divide by peak flux value

    Returns:
        Tuple of (normalized_flux, normalized_flux_err)
    """
    if len(flux) == 0:
        return flux, flux_err

    if method == 'max':
        scale = np.max(np.abs(flux))
    elif method == 'std':
        scale = np.std(flux)
    elif method == 'peak':
        scale = np.max(flux)
    else:
        raise ValueError(f"Unknown method: {method}")

    if scale == 0 or not np.isfinite(scale):
        scale = 1.0

    return flux / scale, flux_err / scale


def compute_snr(flux: np.ndarray, flux_err: np.ndarray) -> np.ndarray:
    """
    Compute signal-to-noise ratio for each observation.

    Args:
        flux: Array of flux values
        flux_err: Array of flux error values

    Returns:
        Array of SNR values
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.abs(flux) / flux_err
        snr = np.where(np.isfinite(snr), snr, 0.0)
    return snr


def filter_low_snr_observations(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    min_snr: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter out observations with low signal-to-noise ratio.

    Args:
        time: Array of time values
        flux: Array of flux values
        flux_err: Array of flux error values
        min_snr: Minimum SNR threshold

    Returns:
        Tuple of filtered (time, flux, flux_err) arrays
    """
    snr = compute_snr(flux, flux_err)
    mask = snr >= min_snr

    return time[mask], flux[mask], flux_err[mask]
