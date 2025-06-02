"""Detect baseline from ECG signal based on lognorm distribution and show signal after baseline wander removal."""
import os
from typing import cast

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.io import loadmat

matplotlib.use("TkAgg")


def detect_baseline(signal: NDArray, frequency: int) -> NDArray:
    """Detects baseline of a signal using scipy library."""
    from scipy.signal import medfilt
    kernel_size = int(1.0 * frequency) | 1
    baseline = medfilt(signal, kernel_size=kernel_size)
    return baseline


def denoise_signal(signal: NDArray, baseline: NDArray) -> NDArray:
    """Remove noises for ECG signal baseline wander removal."""
    return signal - baseline


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


def main():
    """Get signal baseline and remove baseline wander from ECG signal."""

    signal_time_seconds = 5
    ecg_signal, tag, fs = get_example_ecg_signal(seconds=signal_time_seconds)
    baseline = detect_baseline(ecg_signal, fs)
    signal_denoised = denoise_signal(ecg_signal, baseline)

    fig, axs = plt.subplots(
        ncols=2,
        figsize=(12, 4),
        sharey=True, sharex=True,
    )

    t = np.arange(0, signal_time_seconds, 1 / fs)

    signal_ax = cast(Axes, axs[0])
    signal_ax.plot(t, ecg_signal)
    signal_ax.plot(t, baseline, label='Estimated Baseline')
    signal_ax.set_title(f"ECG signal with baseline: {tag}")
    signal_ax.set_xlabel("Time [s]")
    signal_ax.set_ylabel("Amplitude [mV]")
    signal_ax.grid(True)

    distribution_ax = cast(Axes, axs[1])
    distribution_ax.plot(t, signal_denoised)
    distribution_ax.set_title(f"Denoised ECG signal: {tag}")
    distribution_ax.set_xlabel("Time [s]")
    distribution_ax.set_ylabel("Amplitude [mV]")
    distribution_ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
