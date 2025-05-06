"""Wavelet decomposition filters analysis."""
import matplotlib
import pywt
import matplotlib.pyplot as plt
from pywt import Wavelet

matplotlib.use("TkAgg")


def view_wavelet_filters(wavelet: Wavelet):
    """View default wavelet filters."""
    print(f'Analysing {wavelet=}')

    print("Decomposition low-pass filter (h):", wavelet.dec_lo)
    print("Decomposition high-pass filter (g):", wavelet.dec_hi)
    print("Reconstruction low-pass filter (h̃):", wavelet.rec_lo)
    print("Reconstruction high-pass filter (ĝ):", wavelet.rec_hi)


def plot_wavelet(plot_ax, is_basic_wavelet: bool, wavelet: Wavelet):
    """Plot wavelet scaling and base function."""
    print(f'Plotting {wavelet=}')

    phi, psi, x = wavelet.wavefun()

    scaling_plot = plot_ax[0 + 2 * is_basic_wavelet]
    scaling_plot.plot(x, phi)
    scaling_plot.set_title(f"{wavelet_name} Scaling Function φ(x)")
    scaling_plot.set_xlabel("x")
    scaling_plot.set_ylabel("φ(x)")
    scaling_plot.grid(True)

    wavelet_plot = plot_ax[1 + 2 * is_basic_wavelet]
    wavelet_plot.plot(x, psi)
    wavelet_plot.set_title(f"{wavelet_name} Wavelet Function ψ(x)")
    wavelet_plot.set_xlabel("x")
    wavelet_plot.set_ylabel("ψ(x)")
    wavelet_plot.grid(True)


if __name__ == '__main__':
    wavelets_list = {
        'db4': [None],
        'db6': [None],
        'db12': [None],
        'db16': [None],
    }

    fig, axs = plt.subplots(len(wavelets_list), 4, figsize=(3 * len(wavelets_list), 4 * 2))

    for i, wavelet_name in enumerate(wavelets_list):
        wavelet_basic = pywt.Wavelet(wavelet_name)

        view_wavelet_filters(wavelet_basic)
        plot_wavelet(axs[i], False, wavelet_basic)

        wavelet_configured = pywt.Wavelet(wavelet_name)

        # TODO: apply different wavelet decomposition filters

        view_wavelet_filters(wavelet_configured)
        plot_wavelet(axs[i], True, wavelet_configured)

    plt.tight_layout()
    plt.show()
