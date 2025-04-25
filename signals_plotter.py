"""SignalsPlotter class to gather and display signal analysis with wavelets and wavelet transform"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pywt
from numpy.typing import NDArray

matplotlib.use('TkAgg')


class SignalsPlotter:
    """SignalsPlotter class to gather and display signal analysis with wavelets and wavelet transform"""

    def __init__(self) -> None:
        self.signals_set: list[tuple[NDArray, str, NDArray, int, NDArray, NDArray, str]] = []

    def add_signal_with_analysis(
            self, signal_content: tuple[NDArray, str, NDArray, int, NDArray, NDArray, str]
    ) -> None:
        """Add signal to signals_set with its tas, qrs_peaks and frequency and wavelet transform"""
        self.signals_set.append(signal_content)

    def get_max_num_levels(self) -> int:
        """Get maximum number of levels for wavelets decomposition"""
        max_num_levels = 0
        for signal, tag, qrs_peaks, freq, cA, cDs, wavelet in self.signals_set:
            cDs_number = len(cDs)
            if cDs_number > max_num_levels:
                max_num_levels = cDs_number

        return max_num_levels

    def plot_signals(self) -> None:
        """Plot all signals added to signals plotter"""

        squeeze_when_one_signal = False

        num_levels = self.get_max_num_levels()

        fig, axs = plt.subplots(
            num_levels + 3, len(self.signals_set),
            figsize=(6 * len(self.signals_set), 2 * (num_levels + 3)),
            squeeze=squeeze_when_one_signal
        )

        for i, (signal, tag, qrs_peaks, freq, cA, cDs, wavelet) in enumerate(self.signals_set):
            time = np.arange(len(signal)) / freq

            signal_display = signal if signal.ndim == 1 else signal[:, 0]

            axs[0, i].plot(time, signal_display)

            for qrs in qrs_peaks:
                axs[0, i].axvline(x=qrs / freq, color='r', linestyle='--', alpha=0.6)

            axs[0, i].set_title(f"ECG: {tag} with QRS")
            axs[0, i].set_xlabel("Time [s]")
            axs[0, i].set_ylabel("Amplitude [mV]")
            axs[0, i].grid(True)

            for coef_number, coef in enumerate([cA] + cDs):
                level_name = "A" if coef_number == 0 else f"D{num_levels - coef_number}"
                ax = axs[coef_number + 1, i]
                ax.plot(np.arange(len(coef)) / freq, coef)
                ax.set_title(f"Level {level_name} ({wavelet})")
                ax.set_xlabel("Time [s]")
                ax.grid(True)

            wavelet_function = pywt.Wavelet(wavelet)
            phi, psi, x = wavelet_function.wavefun()

            ax_wave = axs[-1, i]
            ax_wave.plot(x, psi)
            ax_wave.set_title(f"Wavelet ψ(t) – {wavelet}")
            ax_wave.set_xlabel("t")
            ax_wave.set_ylabel("ψ(t)")
            ax_wave.grid(True)

        for row_idx in range(1, num_levels + 2):
            y_lims = [axs[row_idx, col].get_ylim() for col in range(len(self.signals_set))]
            common_min = min(lim[0] for lim in y_lims)
            common_max = max(lim[1] for lim in y_lims)

            for col in range(len(self.signals_set)):
                axs[row_idx, col].set_ylim(common_min, common_max)

        plt.tight_layout()
        plt.show()
