"""InverseDWTPlotter class to gather and display signal statistical analysis."""
from collections import Counter

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import NDArray

from plotters.signals_plotter import SignalsPlotter


class SignalStatisticalAnalysisPlotter(SignalsPlotter):
    """InverseDWTPlotter class to gather and display signal statistical analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.signals_set: list[tuple[NDArray, str, NDArray, int, Counter]] = []

    def add_signal_with_analysis(
            self, signal: NDArray, tag: str, qrs_peaks: NDArray, freq: int, data_distribution: Counter
    ):
        """Add signal to signals_set with its tag, qrs_peaks and frequency and wavelet transform."""
        self.signals_set.append((signal, tag, qrs_peaks, freq, data_distribution))

    def compute_plotting(self, **kwargs):
        """Compute plotting signals with its statistical analysis."""

        half_of_signals = int(len(self.signals_set) / 2)

        fig, axs = plt.subplots(
            nrows=2 * 2,
            ncols=half_of_signals,
            figsize=(3 * half_of_signals, 2 * 2 * 2),
            squeeze=False,
            sharex='row',
            sharey='row',
        )

        for signal_number, (signal, tag, qrs_peaks, freq, data_distribution) in enumerate(self.signals_set):
            duration = len(signal) / freq

            bins = 100
            density = True

            if signal_number < half_of_signals:
                signal_plot = axs[0, signal_number]
                dd_plot = axs[1, signal_number]
            else:
                signal_plot = axs[2, signal_number - half_of_signals]
                dd_plot = axs[3, signal_number - half_of_signals]

            self.plot_ecg_signal(signal_plot, signal, duration, freq, tag, qrs_peaks)

            dd_plot.hist(signal, bins=bins, density=density)
