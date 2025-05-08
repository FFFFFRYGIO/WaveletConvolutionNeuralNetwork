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

        fig, axs = plt.subplots(
            nrows=2,
            ncols=len(self.signals_set),
            figsize=(10 * 2, 1 * len(self.signals_set)),
            squeeze=False,
            sharex=True,
        )

        for signal_number, (signal, tag, qrs_peaks, freq, data_distribution) in enumerate(self.signals_set):
            duration = len(signal) / freq

            self.plot_ecg_signal(axs[0, signal_number], signal, duration, freq, tag, qrs_peaks)

            dd_plot = axs[1, signal_number]
