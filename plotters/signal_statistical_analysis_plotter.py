"""InverseDWTPlotter class to gather and display signal statistical analysis."""
import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import NDArray

from plotters.signals_plotter import SignalsPlotter


class SignalStatisticalAnalysisPlotter(SignalsPlotter):
    """InverseDWTPlotter class to gather and display signal statistical analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.signals_set: list[tuple[NDArray, str, NDArray, int, dict, dict]] = []

    def add_signal_with_analysis(
            self, signal: NDArray, tag: str, qrs_peaks: NDArray, freq: int,
            data_dist: dict, detected_dist: dict
    ):
        """Add signal to signals_set with its tag, qrs_peaks and frequency and wavelet transform."""
        self.signals_set.append((signal, tag, qrs_peaks, freq, data_dist, detected_dist))

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

        for signal_number, (signal, tag, qrs_peaks, freq, data_dist, detected_dist) in enumerate(self.signals_set):
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

            # dd_plot.hist(signal, bins=bins, density=density)

            # 1) draw histogram and get the bin‐centers & densities
            counts, bin_edges, _ = dd_plot.hist(
                signal,
                bins=bins,
                density=density,
                alpha=0.6,
                label='Data'
            )
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # 2) fit a quadratic (degree=2) polynomial to (centers, counts)
            coeffs = np.polyfit(centers, counts, deg=2)
            poly = np.poly1d(coeffs)

            # 3) evaluate the polynomial on a fine grid for a smooth line
            xs = np.linspace(centers.min(), centers.max(), 200)
            ys = poly(xs)

            # 4) overlay the fit
            dd_plot.plot(
                xs,
                ys,
                'r-',
                lw=2,
                label=(
                    f'Quad fit:\n'
                    f'{coeffs[0]:.3e}·x² + {coeffs[1]:.3e}·x + {coeffs[2]:.3e}'
                )
            )
            # dd_plot.legend()
