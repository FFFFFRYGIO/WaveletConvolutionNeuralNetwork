"""SignalsPlotter class to gather and display signal analysis with wavelets and wavelet transform."""
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from data_analysis.plotters.signals_plotter import SignalsPlotter


class InverseDWTPlotter(SignalsPlotter):
    """DWTDecompositionPlotter class to gather and display signal analysis with wavelets and wavelet transform."""

    def __init__(self) -> None:
        super().__init__()
        # signals_set element: signal, tag, qrs_peaks, freq, cA, cDs, wavelet, inverse_dwt_list (coeffs, inverse_dwt)
        self.signals_set: list[tuple[
            NDArray, str, NDArray, int, NDArray, NDArray, str, list[tuple[list[str], NDArray]]
        ]] = []

    def add_signal_with_analysis(
        self, signal_content: tuple[NDArray, str, NDArray, int, NDArray, NDArray, str, list[tuple[list[str], NDArray]]]
) -> None:
        """Add signal to signals_set with its tas, qrs_peaks, frequency, wavelet transform and inverse dwt list."""
        self.signals_set.append(signal_content)

    def get_max_num_of_inverse_dwt(self) -> int:
        """Get the maximum number of inverse discrete wavelet transforms."""
        max_num_of_inverse_dwt = 0
        for signal, tag, qrs_peaks, freq, cA, cDs, wavelet, inverse_dwt_list in self.signals_set:
            inverse_dwt_number = len(inverse_dwt_list)
            if inverse_dwt_number > max_num_of_inverse_dwt:
                max_num_of_inverse_dwt = inverse_dwt_number

        return max_num_of_inverse_dwt

    def compute_plotting(self, add_decompositions: bool = False, plot_wavelets: bool = True) -> None:
        """Plot all signals added to signals plotter."""

        squeeze_when_one_signal = False

        num_inverse_dwt = self.get_max_num_of_inverse_dwt()
        num_signals = len(self.signals_set)

        rows_for_decomposition = 0
        if add_decompositions:
            rows_for_decomposition = self.get_max_num_decomposition_levels()

        fig, axs = plt.subplots(
            nrows=1 + rows_for_decomposition + num_inverse_dwt,
            ncols=num_signals + 1 * plot_wavelets,
            figsize=(6 * (num_signals + 1), 2 * (1 + rows_for_decomposition + num_inverse_dwt)),
            squeeze=squeeze_when_one_signal,
            sharex='col',
            sharey='row',
        )

        for i, (signal, tag, qrs_peaks, freq, cA, cDs, wavelet, inverse_dwt_list) in enumerate(self.signals_set):
            duration = len(signal) / freq

            self.plot_ecg_signal(axs[0, i], signal, duration, freq, tag, qrs_peaks)

            if add_decompositions:
                for coef_number, coef in enumerate([cA] + cDs):
                    level_name = "A" if coef_number == 0 else f"D{len(cDs) - coef_number + 1}"
                    ax = axs[coef_number + 1, i]
                    time_coef = np.linspace(0, duration, num=len(coef))
                    ax.plot(time_coef, coef)
                    ax.set_title(f"Level {level_name} ({wavelet})")
                    ax.set_xlabel("Time [s]")
                    ax.set_xlim(0, duration)
                    ax.grid(True)

            for inverse_dwt_number, (coeffs_names, inverse_dwt) in enumerate(inverse_dwt_list):
                ax = axs[inverse_dwt_number + 1 + (len([cA] + cDs) * add_decompositions), i]
                time_inverse_dwt = np.linspace(0, duration, num=len(inverse_dwt))
                ax.plot(time_inverse_dwt, inverse_dwt)

                self.apply_statistics_on_plot(ax, signal, freq=freq, qrs_peaks=qrs_peaks)

                ax.set_title(f"Inverse DWT for {coeffs_names} ({wavelet})")
                ax.set_xlabel("Time [s]")
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
