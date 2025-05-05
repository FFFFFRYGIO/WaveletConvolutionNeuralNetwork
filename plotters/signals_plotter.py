"""SignalsPlotter abstract class for specific signals plotters."""

from abc import ABC, abstractmethod

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")


class SignalsPlotter(ABC):
    """SignalsPlotter class for specific signals plotters that contains main implementations."""

    @abstractmethod
    def compute_plotting(self):
        """Plot all needed elements."""

    def plot_ecg_signal(self):
        """Plot ecg signal."""

    @staticmethod
    def display_plots():
        """Display all signals computed signals."""
        plt.tight_layout()
        plt.show()
