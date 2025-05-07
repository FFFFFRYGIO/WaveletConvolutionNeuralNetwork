"""SignalsPlotter class to gather and display signal analysis with wavelets and wavelet transform."""
import matplotlib.pyplot as plt
import numpy as np
import pywt
from numpy.typing import NDArray

from plotters.signals_plotter import SignalsPlotter


class DWTDecompositionPlotter(SignalsPlotter):
    """DWTDecompositionPlotter class to gather and display signal analysis with wavelets and wavelet transform."""

    def __init__(self) -> None:
        # signals_set element: signal, tag, qrs_peaks, freq, cA, cDs, wavelet
        self.signals_set: list[tuple[NDArray, str, NDArray, int, NDArray, NDArray, str]] = []

    def add_signal_with_analysis(
            self, signal_content: tuple[NDArray, str, NDArray, int, NDArray, NDArray, str]
    ) -> None:
        """Add signal to signals_set with its tas, qrs_peaks, frequency and wavelet transform."""
        self.signals_set.append(signal_content)

    def compute_plotting(self, plot_wavelets: bool = True) -> None:
        """Plot all signals added to signals plotter."""

        squeeze_when_one_signal = False

        num_levels = self.get_max_num_decomposition_levels()
        num_signals = len(self.signals_set)

        fig, axs = plt.subplots(
            nrows=1 + num_levels,
            ncols=num_signals + 1 * plot_wavelets,
            figsize=(6 * (num_signals + 1), 2 * (1 + num_levels)),
            squeeze=squeeze_when_one_signal,
            sharex='col',
            sharey='row',
        )

        for i, (signal, tag, qrs_peaks, freq, cA, cDs, wavelet) in enumerate(self.signals_set):
            duration = len(signal) / freq

            self.plot_ecg_signal(axs[0, i], signal, duration, freq, tag, qrs_peaks)

            for coef_number, coef in enumerate([cA] + cDs):
                level_name = "A" if coef_number == 0 else f"D{len(cDs) - coef_number + 1}"

                if level_name == "A":
                    ax_row_number = 1
                else:
                    ax_row_number = coef_number + 1 + num_levels - len([cA] + cDs)

                ax = axs[ax_row_number, i]
                time_coef = np.linspace(0, duration, num=len(coef))
                ax.plot(time_coef, coef)
                ax.set_title(f"Level {level_name} ({wavelet})")
                # ax.set_xlabel("Time [s]")
                ax.set_xlim(0, duration)
                ax.grid(True)

            if plot_wavelets:
                if i < axs.shape[0]:
                    self.plot_wavelet(axs[i, -1], wavelet, tag)
                else:
                    print(f'Skipped plotting wavelet {wavelet} for {tag}, no space for it')

        if plot_wavelets:
            for i in range(axs.shape[0] - 1, num_signals - 1, -1):
                axs[i, -1].axis('off')
