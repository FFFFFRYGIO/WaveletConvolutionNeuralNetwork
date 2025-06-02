"""Display wavelets functions with their scaling function."""
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import pywt
from matplotlib.axes import Axes

matplotlib.use("TkAgg")


def plot_wavelet(scaling_plot: Axes, wavelet_plot: Axes, wavelet: pywt.Wavelet) -> None:
    """Plot wavelet scaling and base function."""

    phi, psi, x = wavelet.wavefun()

    scaling_plot.plot(x, phi)
    scaling_plot.set_title(f"{wavelet.name} Scaling Function φ(x)")
    scaling_plot.set_xlabel("x")
    scaling_plot.set_ylabel("φ(x)")
    scaling_plot.grid(True)

    wavelet_plot.plot(x, psi)
    wavelet_plot.set_title(f"{wavelet.name} Wavelet Function ψ(x)")
    wavelet_plot.set_xlabel("x")
    wavelet_plot.set_ylabel("ψ(x)")
    wavelet_plot.grid(True)


def main():
    """Display wavelets functions with their scaling function."""

    wavelets_list = ['db4', 'db6', 'sym4']

    fig, axs = plt.subplots(
        nrows=3, ncols=2,
        figsize=(9, 6),
        squeeze=False,
        sharex=True, sharey=True,
    )

    for wavelet_num, wavelet_name in enumerate(wavelets_list):
        wavelet = pywt.Wavelet(wavelet_name)
        scaling_ax = cast(Axes, axs[wavelet_num, 0])
        wavelet_ax = cast(Axes, axs[wavelet_num, 1])
        plot_wavelet(scaling_ax, wavelet_ax, wavelet)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
