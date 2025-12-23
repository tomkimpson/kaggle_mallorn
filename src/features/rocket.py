"""
ROCKET (Random Convolutional Kernel Transform) for time series classification.

A minimal pure-numpy implementation based on:
Dempster et al. (2020) "ROCKET: Exceptionally fast and accurate time series
classification using random convolutional kernels"

Key idea:
1. Generate random convolutional kernels (random length, weights, bias, dilation)
2. Apply each kernel to time series
3. Extract 2 features per kernel: max value and PPV (proportion of positive values)

This gives 2 * num_kernels features that work great with ridge regression.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import signal
import warnings


def _apply_kernel_batch(
    X: np.ndarray,
    weights: np.ndarray,
    bias: float,
    dilation: int,
    padding: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a single dilated kernel to a batch of time series.

    Args:
        X: Time series, shape (n_samples, n_timepoints)
        weights: Kernel weights, shape (kernel_length,)
        bias: Kernel bias
        dilation: Dilation factor
        padding: Padding amount

    Returns:
        max_values: shape (n_samples,)
        ppv_values: shape (n_samples,)
    """
    n_samples, n_timepoints = X.shape
    kernel_length = len(weights)

    # Create dilated kernel
    dilated_length = (kernel_length - 1) * dilation + 1
    dilated_kernel = np.zeros(dilated_length)
    dilated_kernel[::dilation] = weights

    # Pad input
    if padding > 0:
        X_padded = np.pad(X, ((0, 0), (padding, padding)), mode='constant', constant_values=0)
    else:
        X_padded = X

    # Convolve (using 'valid' mode since we already padded)
    # scipy.signal.convolve is slow, use np.apply_along_axis with np.convolve
    def convolve_row(row):
        return np.convolve(row, dilated_kernel[::-1], mode='valid') + bias

    output = np.apply_along_axis(convolve_row, 1, X_padded)

    # Handle edge case where output is empty
    if output.shape[1] == 0:
        return np.zeros(n_samples), np.zeros(n_samples)

    # Extract features
    max_values = np.max(output, axis=1)
    ppv_values = np.mean(output > 0, axis=1)

    return max_values, ppv_values


class MiniROCKET:
    """
    Minimal ROCKET implementation for time series classification.

    Usage:
        rocket = MiniROCKET(num_kernels=10000)
        rocket.fit(X_train)  # X_train shape: (n_samples, n_timepoints)
        X_train_features = rocket.transform(X_train)  # shape: (n_samples, 20000)
        X_test_features = rocket.transform(X_test)
    """

    def __init__(
        self,
        num_kernels: int = 10000,
        max_kernel_length: int = 9,
        seed: int = 42,
    ):
        """
        Initialize ROCKET.

        Args:
            num_kernels: Number of random kernels to generate
            max_kernel_length: Maximum kernel length (typically 7 or 9)
            seed: Random seed for reproducibility
        """
        self.num_kernels = num_kernels
        self.max_kernel_length = max_kernel_length
        self.seed = seed
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'MiniROCKET':
        """
        Generate random kernels based on input time series length.

        Args:
            X: Time series array of shape (n_samples, n_timepoints)

        Returns:
            self
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_timepoints = X.shape[1]
        rng = np.random.RandomState(self.seed)

        # Candidate kernel lengths (odd numbers)
        candidate_lengths = np.array([7, 9])

        # Generate random kernel parameters
        self.lengths = rng.choice(candidate_lengths, size=self.num_kernels)

        # Weights: random from {-1, 0, 1} normalized
        self.weights = []
        for i in range(self.num_kernels):
            length = self.lengths[i]
            w = rng.choice([-1.0, 0.0, 1.0], size=length)
            w = w - w.mean()  # Zero-mean
            self.weights.append(w)

        # Biases: uniform in a reasonable range
        self.biases = rng.uniform(-1, 1, size=self.num_kernels)

        # Dilations: exponential distribution
        max_exponent = np.log2(max(1, (n_timepoints - 1) / (self.max_kernel_length - 1)))
        self.dilations = np.floor(2 ** rng.uniform(0, max_exponent, size=self.num_kernels)).astype(np.int32)
        self.dilations = np.maximum(1, self.dilations)

        # Paddings: either 0 or enough for "same" convolution
        self.paddings = np.zeros(self.num_kernels, dtype=np.int32)
        for i in range(self.num_kernels):
            if rng.random() > 0.5:
                self.paddings[i] = ((self.lengths[i] - 1) * self.dilations[i]) // 2

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform time series to ROCKET features.

        Args:
            X: Time series array of shape (n_samples, n_timepoints)

        Returns:
            Features array of shape (n_samples, num_kernels * 2)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        features = np.zeros((n_samples, self.num_kernels * 2))

        X = X.astype(np.float64)

        for k in range(self.num_kernels):
            max_vals, ppv_vals = _apply_kernel_batch(
                X,
                self.weights[k],
                self.biases[k],
                self.dilations[k],
                self.paddings[k],
            )
            features[:, k * 2] = max_vals
            features[:, k * 2 + 1] = ppv_vals

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class MultiChannelROCKET:
    """
    ROCKET for multi-channel time series (e.g., 6 photometric bands).

    Applies ROCKET to each channel and concatenates features.
    """

    def __init__(
        self,
        num_kernels_per_channel: int = 1000,
        max_kernel_length: int = 9,
        seed: int = 42,
    ):
        """
        Initialize multi-channel ROCKET.

        Args:
            num_kernels_per_channel: Kernels per channel (total features = n_channels * kernels * 2)
            max_kernel_length: Maximum kernel length
            seed: Random seed
        """
        self.num_kernels_per_channel = num_kernels_per_channel
        self.max_kernel_length = max_kernel_length
        self.seed = seed
        self.rockets = {}
        self.n_channels = None

    def fit(self, X: np.ndarray) -> 'MultiChannelROCKET':
        """
        Fit ROCKET for each channel.

        Args:
            X: Multi-channel time series, shape (n_samples, n_channels, n_timepoints)

        Returns:
            self
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (samples, channels, time), got {X.ndim}D")

        self.n_channels = X.shape[1]

        for c in range(self.n_channels):
            self.rockets[c] = MiniROCKET(
                num_kernels=self.num_kernels_per_channel,
                max_kernel_length=self.max_kernel_length,
                seed=self.seed + c,
            )
            self.rockets[c].fit(X[:, c, :])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform multi-channel time series.

        Args:
            X: Multi-channel time series, shape (n_samples, n_channels, n_timepoints)

        Returns:
            Features, shape (n_samples, n_channels * num_kernels_per_channel * 2)
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array, got {X.ndim}D")

        all_features = []
        for c in range(self.n_channels):
            features = self.rockets[c].transform(X[:, c, :])
            all_features.append(features)

        return np.hstack(all_features)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


def test_rocket():
    """Quick test of ROCKET implementation."""
    np.random.seed(42)
    n_samples = 50
    n_timepoints = 100

    # Single channel
    X = np.random.randn(n_samples, n_timepoints)
    rocket = MiniROCKET(num_kernels=100)
    features = rocket.fit_transform(X)
    print(f"Single channel: {X.shape} -> {features.shape}")
    assert features.shape == (n_samples, 200), f"Got {features.shape}"

    # Multi-channel
    X_multi = np.random.randn(n_samples, 6, n_timepoints)
    mc_rocket = MultiChannelROCKET(num_kernels_per_channel=50)
    features_multi = mc_rocket.fit_transform(X_multi)
    print(f"Multi-channel: {X_multi.shape} -> {features_multi.shape}")
    assert features_multi.shape == (n_samples, 6 * 50 * 2), f"Got {features_multi.shape}"

    print("ROCKET tests passed!")


if __name__ == '__main__':
    test_rocket()
