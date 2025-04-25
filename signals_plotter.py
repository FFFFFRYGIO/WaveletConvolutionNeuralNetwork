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
        self.num_levels = 2

    def add_signal_with_analysis(
            self, signal_content: tuple[NDArray, str, NDArray, int, NDArray, NDArray, str]
    ) -> None:
        """Add signal to signals_set with its tas, qrs_peaks and frequency and wavelet transform"""
        self.signals_set.append(signal_content)

    def plot_signals(self) -> None:
        """Plot all signals added to signals plotter"""

        squeeze_when_one_signal = False

        fig, axs = plt.subplots(
            4, len(self.signals_set),
            figsize=(10 * len(self.signals_set), 3 * (self.num_levels + 2)),
            squeeze=squeeze_when_one_signal
        )

        for i, (signal, tag, qrs_peaks, freq, cA, cD, wavelet) in enumerate(self.signals_set):
            time = np.arange(len(signal)) / freq

            if signal.ndim == 1:
                signal_display = signal
            else:
                signal_display = signal[:, 0]

            axs[0, i].plot(time, signal_display)

            for qrs in qrs_peaks:
                axs[0, i].axvline(x=qrs / freq, color='r', linestyle='--', alpha=0.6)

            axs[0, i].set_title(f"ECG: {tag} with QRS")
            axs[0, i].set_xlabel("Time [s]")
            axs[0, i].set_ylabel("Amplitude [mV]")
            axs[0, i].grid(True)

            for coef_number, coef in enumerate([cA, cD]):
                level_name = "A" if coef_number == 0 else f"D{self.num_levels - coef_number}"
                axs[coef_number + 1, i].plot(np.arange(len(coef)) / freq, coef)
                axs[coef_number + 1, i].set_title(f"Level {level_name} ({wavelet})")
                axs[coef_number + 1, i].set_xlabel("Time [s]")
                axs[coef_number + 1, i].grid(True)

            wavelet_function = pywt.Wavelet(wavelet)
            phi, psi, x = wavelet_function.wavefun()

            axs[self.num_levels + 1, i].plot(x, psi)
            axs[self.num_levels + 1, i].set_title(f"Wavelet ψ(t) – {wavelet}")
            axs[self.num_levels + 1, i].grid(True)
            axs[self.num_levels + 1, i].set_xlabel("t")
            axs[self.num_levels + 1, i].set_ylabel("ψ(t)")

        plt.tight_layout()
        plt.show()
