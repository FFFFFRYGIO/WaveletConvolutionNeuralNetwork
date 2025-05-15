"""Class that plots all signals analysis content for comparison."""
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

matplotlib.use("TkAgg")


class DWTExperimentsPlotter:
    """SignalsPlotter class for specific signals plotters that contains main implementations."""

    def __init__(self, signals_contents: list[dict], signals_frequency: int) -> None:
        matplotlib.use("TkAgg")
        self.signals_contents = signals_contents
        self.fs: int = signals_frequency
        self.larges_dwt = max([len(signals_content['dwt']) for signals_content in signals_contents])
        self.larges_idwt = max([len(signals_content['idwt']) for signals_content in signals_contents])
        self.fig, self.axs = None, None

    def compute_plotting(self):
        """Plot signal with his specific analysis."""

        self.fig, self.axs = plt.subplots(
            nrows=1 + self.larges_dwt + self.larges_idwt,
            ncols=len(self.signals_contents),
            figsize=(3 * len(self.signals_contents), 2 * (1 + 1 + self.larges_dwt + self.larges_idwt)),
            squeeze=False,
            sharex=True, sharey='row',
        )

        for signal_content_number, signal_content in enumerate(self.signals_contents):
            tag, signal = signal_content['signal']
            time_signal = np.linspace(0, len(signal) / self.fs, num=len(signal))
            self.plot_ecg_signal(signal_content_number, signal, tag, time_signal)

    def plot_ecg_signal(self, signal_content_number: int, signal: NDArray, tag: str, time_signal: NDArray) -> None:
        """Plot ecg signal with qrs_peaks."""
        signal_plot = self.axs[0, signal_content_number]
        signal_plot.plot(time_signal, signal)
        signal_plot.set_title(f"ECG: {tag} with QRS (l={len(signal)})")
        signal_plot.set_xlabel("Time [s]")
        signal_plot.set_ylabel("Amplitude [mV]")
        signal_plot.grid(True)

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
