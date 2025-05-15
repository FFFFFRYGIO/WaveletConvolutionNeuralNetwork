"""Run and display signal after different normalization processes."""
import os
from typing import cast

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.io import loadmat

matplotlib.use("TkAgg")


def get_example_ecg_signal(seconds: int | None = None) -> tuple[NDArray, str, int]:
    """Get example ECG signal from ECGData from Physionet"""

    data_source = os.getenv('DATA_SOURCE')
    mat_data = loadmat(data_source)
    raw_data = mat_data['ECGData']
    source_signals, labels = raw_data[0, 0]

    # Get healthy signal based on documentation
    nsr_signal = source_signals[96 + 30]

    # Get signal frequency based on documentation
    frequency = int(os.getenv('ECGDATA_FREQUENCY'))

    if seconds is not None:
        nsr_signal = nsr_signal[:seconds * frequency]

    return nsr_signal, 'NSR', frequency


def normalize_signal(signal: NDArray, normalization_mode: str) -> NDArray:
    """Normalize ECG signal my max-abs (from -1 to 1) or minmax (from 0 to 1)."""

    match normalization_mode:
        case 'max-abs':
            max_peak = np.max(np.abs(signal))

            normalized_signal = signal / max_peak if max_peak else signal

        case 'minmax':
            signal_max = max(signal)
            signal_min = min(signal)

            normalized_signal = (signal - signal_min) / (signal_max - signal_min)

        case 'raw':
            normalized_signal = signal

        case _:
            raise ValueError(f"{normalization_mode=} is not a valid normalization mode")

    return normalized_signal


def main():
    """Run different ECG signal normalizations."""

    signal_time_seconds = 5

    ecg_signal, tag, fs = get_example_ecg_signal(seconds=signal_time_seconds)

    fig, axs = plt.subplots(
        ncols=3,
        figsize=(12, 4),
        sharey=True, sharex=True,
    )
    t = np.arange(0, signal_time_seconds, 1 / fs)

    signal_normalization_operations = ['raw', 'minmax', 'max-abs']
    for i, normalization_operation in enumerate(signal_normalization_operations):
        signal = normalize_signal(ecg_signal, normalization_operation)

        signal_ax = cast(Axes, axs[i])
        signal_ax.plot(t, signal)
        signal_ax.set_title(f"ECG signal: {tag} norm. mode: {normalization_operation}")
        signal_ax.set_xlabel("Time [s]")
        signal_ax.set_ylabel("Amplitude [mV]")
        signal_ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
