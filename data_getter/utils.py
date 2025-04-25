"""Help functions for all data getters."""
from numpy.core.multiarray import ndarray
from numpy.typing import NDArray
from scipy.signal import find_peaks


def get_signal_subset(signal: ndarray, signal_frequency: int, seconds: int | None = None) -> ndarray:
    """Get specified time for signal."""

    if seconds:
        duration_samples = int(seconds * signal_frequency)
        signal = signal[:duration_samples]

    return signal


def get_qrs_peaks(signal: NDArray) -> NDArray:
    """Get qrs peaks times"""

    # Main setup
    qrs_height_factor = 1.0
    qrs_min_distance = 0.1

    height_thresh = signal.mean() + qrs_height_factor * signal.std()
    min_dist_samples = int(qrs_min_distance * signal.shape[0])

    peaks, _ = find_peaks(
        signal,
        height=height_thresh,
        distance=min_dist_samples
    )

    return peaks
