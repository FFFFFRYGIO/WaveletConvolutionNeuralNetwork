"""Help functions for all data getters."""
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


def get_signal_subset(signal: NDArray, signal_frequency: int, seconds: int | None = None) -> NDArray:
    """Get specified time for signal."""

    if seconds:
        duration_samples = int(seconds * signal_frequency)
        signal = signal[:duration_samples]

    return signal


def get_qrs_peaks(
        signal: NDArray,
        qrs_locs: NDArray | None = None,
        seconds: int | None = None,
        signal_frequency: int | None = None
) -> NDArray:
    """Get qrs peaks times"""

    if qrs_locs is not None and len(qrs_locs) > 0:
        if seconds is None or signal_frequency is None:
            raise ValueError(f"{seconds=} and {signal_frequency=} must be provided when qrs_locs is used")

        duration_samples = int(seconds * signal_frequency)
        qrs_peaks_in_range = [loc for loc in qrs_locs if loc < duration_samples]
        return np.array(qrs_peaks_in_range, dtype=float)

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
