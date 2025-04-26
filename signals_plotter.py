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
        num_signals = len(self.signals_set)

        fig, axs = plt.subplots(
            num_levels + 3, num_signals,
            figsize=(6 * num_signals, 2 * (num_levels + 3)),
            squeeze=squeeze_when_one_signal
        )

        for i, (signal, tag, qrs_peaks, freq, cA, cDs, wavelet) in enumerate(self.signals_set):
            duration = len(signal) / freq
            time_signal = np.linspace(0, duration, num=len(signal))

            signal_display = signal if signal.ndim == 1 else signal[:, 0]

            signal_ax = axs[0, i]
            signal_ax.plot(time_signal, signal_display)
            signal_ax.set_title(f"ECG: {tag} with QRS (len={len(signal)})")
            signal_ax.set_xlabel("Time [s]")
            signal_ax.set_ylabel("Amplitude [mV]")
            signal_ax.set_xlim(0, duration)
            signal_ax.grid(True)
            for qrs in qrs_peaks:
                signal_ax.axvline(x=qrs / freq, color='r', linestyle='--', alpha=0.6)

            for coef_number, coef in enumerate([cA] + cDs):
                level_name = "A" if coef_number == 0 else f"D{len(cDs) - coef_number + 1}"
                ax = axs[coef_number + 1, i]
                time_coef = np.linspace(0, duration, num=len(coef))
                ax.plot(time_coef, coef)
                ax.set_title(f"Level {level_name} ({wavelet})")
                ax.set_xlabel("Time [s]")
                ax.set_xlim(0, duration)
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
            y_lims = [axs[row_idx, col].get_ylim() for col in range(num_signals)]
            common_min = min(lim[0] for lim in y_lims)
            common_max = max(lim[1] for lim in y_lims)
            for col in range(num_signals):
                axs[row_idx, col].set_ylim(common_min, common_max)

        plt.tight_layout()
        plt.show()
