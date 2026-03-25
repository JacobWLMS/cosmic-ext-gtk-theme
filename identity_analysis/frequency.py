"""Frequency-domain utilities for latent analysis."""

import numpy as np
from typing import Optional


def fft_2d_per_channel(latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2D FFT of a latent tensor per channel.

    Args:
        latent: Array of shape [1, C, H, W] or [C, H, W]

    Returns:
        (magnitude, phase) each of shape [C, H, W]
    """
    if latent.ndim == 4:
        latent = latent[0]

    C, H, W = latent.shape
    magnitude = np.zeros_like(latent)
    phase = np.zeros_like(latent)

    for c in range(C):
        fft = np.fft.fft2(latent[c])
        fft_shifted = np.fft.fftshift(fft)
        magnitude[c] = np.abs(fft_shifted)
        phase[c] = np.angle(fft_shifted)

    return magnitude, phase


def ifft_2d_per_channel(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Inverse FFT from magnitude and phase.

    Args:
        magnitude: [C, H, W] frequency magnitudes
        phase: [C, H, W] frequency phases

    Returns:
        Reconstructed latent of shape [1, C, H, W]
    """
    C, H, W = magnitude.shape
    result = np.zeros_like(magnitude)

    for c in range(C):
        fft_shifted = magnitude[c] * np.exp(1j * phase[c])
        fft = np.fft.ifftshift(fft_shifted)
        result[c] = np.real(np.fft.ifft2(fft))

    return result[np.newaxis]


def frequency_band_mask(
    H: int,
    W: int,
    band: str = "low",
    cutoff_low: float = 0.0,
    cutoff_high: float = 0.33,
) -> np.ndarray:
    """Create a frequency band mask for 2D FFT.

    Args:
        H, W: Spatial dimensions
        band: One of "low", "mid", "high", or "custom"
        cutoff_low: Lower normalized frequency cutoff (0-1, fraction of max freq)
        cutoff_high: Upper normalized frequency cutoff (0-1, fraction of max freq)

    Returns:
        Mask of shape [H, W] with values 0 or 1
    """
    if band == "low":
        cutoff_low, cutoff_high = 0.0, 0.33
    elif band == "mid":
        cutoff_low, cutoff_high = 0.33, 0.66
    elif band == "high":
        cutoff_low, cutoff_high = 0.66, 1.0
    # "custom" uses the provided cutoffs

    cy, cx = H // 2, W // 2
    max_radius = np.sqrt(cy**2 + cx**2)

    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    normalized = dist / max_radius

    mask = ((normalized >= cutoff_low) & (normalized <= cutoff_high)).astype(np.float32)
    return mask


def apply_frequency_mask(
    latent: np.ndarray,
    band: str = "low",
    cutoff_low: float = 0.0,
    cutoff_high: float = 0.33,
) -> np.ndarray:
    """Apply a frequency band mask to a latent and return the filtered result.

    Args:
        latent: [1, C, H, W] or [C, H, W]

    Returns:
        Filtered latent of same shape as input
    """
    squeeze = latent.ndim == 3
    if latent.ndim == 4:
        latent = latent[0]

    C, H, W = latent.shape
    mask = frequency_band_mask(H, W, band, cutoff_low, cutoff_high)
    result = np.zeros_like(latent)

    for c in range(C):
        fft = np.fft.fft2(latent[c])
        fft_shifted = np.fft.fftshift(fft)
        fft_masked = fft_shifted * mask
        fft_unshifted = np.fft.ifftshift(fft_masked)
        result[c] = np.real(np.fft.ifft2(fft_unshifted))

    if squeeze:
        return result
    return result[np.newaxis]


def spatial_windowed_fft(
    latent: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply FFT within a spatial region (e.g., face crop) of the latent.

    Args:
        latent: [1, C, H, W] or [C, H, W]
        bbox: (y1, x1, y2, x2) in latent space coordinates

    Returns:
        (magnitude, phase) of the cropped region, each [C, crop_H, crop_W]
    """
    if latent.ndim == 4:
        latent = latent[0]

    y1, x1, y2, x2 = bbox
    crop = latent[:, y1:y2, x1:x2].copy()
    return fft_2d_per_channel(crop)


def compute_frequency_band_energy(
    latent: np.ndarray,
    n_bands: int = 10,
) -> np.ndarray:
    """Compute energy in each frequency band per channel.

    Args:
        latent: [1, C, H, W] or [C, H, W]
        n_bands: Number of frequency bands

    Returns:
        Array of shape [C, n_bands] with energy values
    """
    if latent.ndim == 4:
        latent = latent[0]

    C, H, W = latent.shape
    energies = np.zeros((C, n_bands))

    for band_idx in range(n_bands):
        low = band_idx / n_bands
        high = (band_idx + 1) / n_bands
        mask = frequency_band_mask(H, W, "custom", low, high)

        for c in range(C):
            fft = np.fft.fft2(latent[c])
            fft_shifted = np.fft.fftshift(fft)
            mag = np.abs(fft_shifted)
            energies[c, band_idx] = np.sum(mag * mask)

    return energies
