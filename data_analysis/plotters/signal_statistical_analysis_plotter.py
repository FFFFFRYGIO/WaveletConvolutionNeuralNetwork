import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import NDArray
from scipy.stats import lognorm

from data_analysis.plotters.signals_plotter import SignalsPlotter


class SignalStatisticalAnalysisPlotter(SignalsPlotter):
    """InverseDWTPlotter class to gather and display signal statistical analysis."""

    def __init__(self) -> None:
        super().__init__()
        # Each entry: (signal, tag, qrs_peaks, freq, data_dist, detected_dist)
        self.signals_set: list[tuple[NDArray, str, NDArray, int, dict, dict]] = []

    def add_signal_with_analysis(
        self, signal: NDArray, tag: str, qrs_peaks: NDArray, freq: int,
        data_dist: dict, detected_dist: dict
    ):
        """Add signal with its statistical analysis results."""
        self.signals_set.append((signal, tag, qrs_peaks, freq, data_dist, detected_dist))

    def compute_plotting(self, add_detected_distribution: bool = True):
        """Plot ECG signals and overlay histogram with fitted distributions."""

        half_of_signals = len(self.signals_set) // 2

        fig, axs = plt.subplots(
            nrows=2 * 2,
            ncols=half_of_signals,
            figsize=(3 * half_of_signals, 2 * 2 * 2),
            squeeze=False,
            # sharex='row',
            # sharey='row',
        )

        for signal_number, (signal, tag, qrs_peaks, freq, data_dist, detected_dist) in enumerate(self.signals_set):
            duration = len(signal) / freq

            bins = 100
            density = True

            row = 0 if signal_number < half_of_signals else 2
            col = signal_number if signal_number < half_of_signals else signal_number - half_of_signals

            ax_signal = axs[row, col]
            self.plot_ecg_signal(ax_signal, signal, duration, freq, tag, qrs_peaks)

            ax_hist = axs[row + 1, col]

            counts, bin_edges, _ = ax_hist.hist(
                signal,
                bins=bins,
                density=density,
                alpha=0.6,
                label='Data'
            )
            ax_hist.set_title(f'{tag} Distribution')

            if add_detected_distribution:
                centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                xs = np.linspace(centers.min(), centers.max(), 200)

                name = detected_dist.get('name')
                params = detected_dist.get('params', ())
                if name == 'lognorm' and len(params) >= 3:
                    shape, loc, scale = params
                    pdf_vals = lognorm.pdf(xs, shape, loc=loc, scale=scale)
                    ax_hist.plot(
                        xs,
                        pdf_vals,
                        '--',
                        lw=2,
                        label=f'{name} fit'
                    )

        plt.tight_layout()
        plt.show()
