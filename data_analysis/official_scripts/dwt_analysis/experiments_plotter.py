"""Class that plots all signals analysis content for comparison."""
from typing import cast

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from data_analysis.official_scripts.dwt_analysis.ecg_signal import ECGSignalContent

matplotlib.use("TkAgg")


class DWTExperimentsPlotter:
    """SignalsPlotter class for specific signals plotters that contains main implementations."""

    def __init__(self, signals_contents: list[ECGSignalContent], signals_frequency: int) -> None:
        matplotlib.use("TkAgg")
        self.signals_contents = signals_contents
        self.signals_length = len(self.signals_contents)
        self.fs: int = signals_frequency
        self.largest_dwt = max([len(signals_content.dwt_decomposition) for signals_content in signals_contents])
        self.largest_idwt = max([len(signals_content.dwt_reconstructions) for signals_content in signals_contents])
        self.axs_rows = 1 + self.largest_dwt + self.largest_idwt
        self.axs_cols = self.signals_length
        self.fig, self.axs = None, None

    def compute_plotting(self):
        """Plot signal with his specific analysis."""

        self.fig, self.axs = plt.subplots(
            nrows=self.axs_rows,
            ncols=self.axs_cols,
            figsize=(3 * self.axs_cols, 1.5 * self.axs_rows),
            squeeze=False,
            sharex=True, sharey='row',
        )

        for sig_content_number, sig_content in enumerate(self.signals_contents):
            signal = sig_content.signal
            duration = len(signal) / self.fs

            time_signal = np.linspace(0, duration, num=len(signal))
            sig_plot = cast(Axes, self.axs[0, sig_content_number])
            sig_plot.plot(time_signal, signal)
            sig_title = f"S: {sig_content.tag} {sig_content.mode}, {sig_content.wavelet.name} (len={len(signal)})"
            sig_plot.set_title(sig_title)
            sig_plot.set_xlabel("Time [s]")
            sig_plot.set_ylabel("Amplitude [mV]")
            sig_plot.grid(True)

            for coeff_number, (coeff_name, coeff) in enumerate(sig_content.dwt_decomposition.items()):
                time_coeff = np.linspace(0, duration, num=len(coeff))
                dwt_plot = cast(Axes, self.axs[coeff_number + 1, sig_content_number])
                dwt_plot.plot(time_coeff, coeff)
                dwt_plot.set_title(f"Level {coeff_name} (len={len(coeff)})")
                dwt_plot.set_xlabel("Time [s]")
                dwt_plot.set_xlim(0, duration)
                dwt_plot.grid(True)

            for idwt_number, (coeffs_names, inverse_dwt) in enumerate(sig_content.dwt_reconstructions):
                time_inverse_dwt = np.linspace(0, duration, num=len(inverse_dwt))
                inverse_dwt_plot = cast(Axes, self.axs[idwt_number + self.largest_dwt + 1, sig_content_number])
                inverse_dwt_plot.plot(time_inverse_dwt, inverse_dwt)
                inverse_dwt_plot.set_title(f"IDWT {coeffs_names} (len={len(inverse_dwt)})")
                inverse_dwt_plot.set_xlabel("Time [s]")
                inverse_dwt_plot.set_xlim(0, duration)
                inverse_dwt_plot.grid(True)

    @staticmethod
    def display_plots(maximized: bool = True):
        """Display all signals computed signals."""
        plt.tight_layout()
        if maximized:
            mgr = plt.get_current_fig_manager()
            win = getattr(mgr, "window", None)
            win.state("zoomed")  # TkAgg
        plt.show()
