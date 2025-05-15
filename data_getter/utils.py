"""Help functions for all data getters."""
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


def get_signal_subset(
        signal: NDArray, signal_frequency: int, seconds: int | None = None, sample_shift_seconds: int | None = None
) -> NDArray:
    """Get specified time for signal."""

    sample_shifting = 0
    if sample_shift_seconds:
        sample_shifting = sample_shift_seconds * signal_frequency

    if seconds:
        duration_samples = int(seconds * signal_frequency)
        signal = signal[sample_shifting: duration_samples + sample_shifting]

    return signal


def get_qrs_peaks(
        signal: NDArray,
        signal_frequency: int,
        qrs_locs: NDArray | None = None,
        seconds: int | None = None,
) -> NDArray:
    """Get qrs peaks times."""

    if qrs_locs is not None and len(qrs_locs) > 0:
        if seconds is None or signal_frequency is None:
            raise ValueError(f"{seconds=} and {signal_frequency=} must be provided when qrs_locs is used")

        duration_samples = int(seconds * signal_frequency)
        qrs_peaks_in_range = [loc for loc in qrs_locs if loc < duration_samples]
        return np.array(qrs_peaks_in_range, dtype=float)

    peaks, _ = find_peaks(
        signal,
        height=0.5,
        distance=int(signal_frequency / 10)
    )

    return peaks


def normalize_signal(signal: NDArray, normalization_mode: str = 'max-abs') -> NDArray:
    """Normalize ECG signal my max-abs (from -1 to 1) or minmax (from 0 to 1)."""

    match normalization_mode:
        case 'max-abs':
            max_peak = np.max(np.abs(signal))

            normalized_signal = signal / max_peak if max_peak else signal

        case 'minmax':
            signal_max = max(signal)
            signal_min = min(signal)

            normalized_signal = (signal - signal_min) / (signal_max - signal_min)

        case None:
            normalized_signal = signal

        case _:
            raise ValueError(f"{normalization_mode=} is not a valid normalization mode")

    return normalized_signal
