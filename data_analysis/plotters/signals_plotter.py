"""SignalsPlotter abstract class for specific signals plotters."""

from abc import ABC, abstractmethod

import matplotlib
import numpy as np
import pywt
from matplotlib import pyplot as plt
from numpy._typing import NDArray

matplotlib.use("TkAgg")


class SignalsPlotter(ABC):
    """SignalsPlotter class for specific signals plotters that contains main implementations."""

    def __init__(self) -> None:
        matplotlib.use("TkAgg")

    @abstractmethod
    def add_signal_with_analysis(self):
        """Add signal to signals_set with its tag, qrs_peaks and frequency and wavelet transform."""

    @abstractmethod
    def compute_plotting(self, signal_content: tuple):
        """Plot signal with his specific analysis."""

    def get_max_num_decomposition_levels(self) -> int:
        """Get the maximum number of levels for wavelets decomposition."""
        max_num_levels = 0
        for signal_data in self.signals_set:
            cDs = signal_data[5]
            cDs_number = len(cDs)
            if cDs_number > max_num_levels:
                max_num_levels = cDs_number

        return max_num_levels + 1

    @staticmethod
    def apply_statistics_on_plot(
            plot_ax, signal: NDArray,
            duration: float | None = None,
            freq: int | None = None,
            qrs_peaks: NDArray | None = None,
            color_for_range: bool = False,
            hide_values_from_range: bool = False,
    ):
        """For given signal calculate statistics and display them on a given plot."""

        mean_val = np.mean(signal)
        std_val = np.std(signal)

        plot_ax.axhline(mean_val, color='k', linestyle='--', linewidth=1, label='Mean')
        plot_ax.axhline(mean_val + std_val, color='r', linestyle='-.', linewidth=1, label='+1 STD')
        plot_ax.axhline(mean_val - std_val, color='r', linestyle='-.', linewidth=1, label='−1 STD')

        if color_for_range:
            plot_ax.fill_between(
                signal, mean_val - std_val, mean_val + std_val,
                color='gray', alpha=0.3, label='±1 STD region'
            )

        if hide_values_from_range:
            if not (duration and freq):
                raise ValueError(f'Both values needed to mask the signal {duration=}, {freq=}')

            time_signal = np.linspace(0, duration, num=len(signal))
            mask = (signal > mean_val + std_val) | (signal < mean_val - std_val)
            masked_signal = np.where(mask, signal, np.nan)
            plot_ax.plot(time_signal, masked_signal)

        if qrs_peaks is not None:
            if not freq:
                raise ValueError(f'Missing freq value, {freq=}')

            for qrs in qrs_peaks:
                plot_ax.axvline(x=qrs / freq, color='r', linestyle='--', alpha=0.1)

    @staticmethod
    def plot_ecg_signal(plot_ax, signal: NDArray, duration: float, freq: int, tag: str, qrs_peaks: NDArray):
        """Plot ecg signal with qrs_peaks."""

        time_signal = np.linspace(0, duration, num=len(signal))

        plot_ax.plot(time_signal, signal)
        plot_ax.set_title(f"ECG: {tag} with QRS (l={len(signal)})")
        plot_ax.set_xlabel("Time [s]")
        plot_ax.set_ylabel("Amplitude [mV]")
        plot_ax.set_xlim(0, duration)
        plot_ax.grid(True)
        for qrs in qrs_peaks:
            plot_ax.axvline(x=qrs / freq, color='r', linestyle='--', alpha=0.1)

    @staticmethod
    def plot_wavelet(plot_ax, wavelet: str, tag: str):
        """Plot wavelet."""

        wavelet_function = pywt.Wavelet(wavelet)
        phi, psi, x = wavelet_function.wavefun()

        plot_ax.plot(x, psi)
        plot_ax.set_title(f"Wavelet for {tag} ψ(t) – {wavelet}")
        plot_ax.set_xlabel("t")
        plot_ax.set_ylabel("ψ(t)")
        plot_ax.grid(True)

    @staticmethod
    def display_plots(maximized: bool = True):
        """Display all signals computed signals."""
        plt.tight_layout()

        if maximized:
            mgr = plt.get_current_fig_manager()
            win = getattr(mgr, "window", None)

            if win is not None and hasattr(win, "showMaximized"):
                win.showMaximized()
            elif hasattr(mgr, "full_screen_toggle"):  # Fallback for Qt’s new API
                mgr.full_screen_toggle()
            elif win is not None and hasattr(win, "state"):  # TkAgg
                win.state("zoomed")
            elif win is not None and hasattr(win, "maximize"):  # GTK, WX, etc.
                win.maximize()

        plt.show()
