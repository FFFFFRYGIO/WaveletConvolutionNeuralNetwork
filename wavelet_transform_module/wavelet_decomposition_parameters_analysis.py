"""Wavelet decomposition filters analysis."""
import matplotlib
import pywt
import matplotlib.pyplot as plt
from pywt import Wavelet, orthogonal_filter_bank

matplotlib.use("TkAgg")


def view_wavelet_filters(wavelet: Wavelet):
    """View default wavelet filters."""
    print(f'Analysing {wavelet=}')

    print("Decomposition low-pass filter (h):", wavelet.dec_lo)
    print("Decomposition high-pass filter (g):", wavelet.dec_hi)
    print("Reconstruction low-pass filter (h̃):", wavelet.rec_lo)
    print("Reconstruction high-pass filter (ĝ):", wavelet.rec_hi)


def plot_wavelet(plot_ax, ord_number, wavelet_name, wavelet_basic: Wavelet, wavelet_custom: Wavelet):
    """Plot wavelet scaling and base function."""
    print(f'Plotting\n  BASIC: {wavelet_basic}\n  CUSTOM: {wavelet_custom}')

    for row, wavelet in enumerate((wavelet_basic, wavelet_custom)):
        phi, psi, x = wavelet.wavefun()[:3]

        scaling_plot = plot_ax[row, 0 + 2 * ord_number]
        scaling_plot.plot(x, phi)
        scaling_plot.set_title(f"{wavelet_name} Scaling Function φ(x)")
        scaling_plot.set_xlabel("x")
        scaling_plot.set_ylabel("φ(x)")
        scaling_plot.grid(True)

        wavelet_plot = plot_ax[row, 1 + 2 * ord_number]
        wavelet_plot.plot(x, psi)
        wavelet_plot.set_title(f"{wavelet_name} Wavelet Function ψ(x)")
        wavelet_plot.set_xlabel("x")
        wavelet_plot.set_ylabel("ψ(x)")
        wavelet_plot.grid(True)


if __name__ == '__main__':
    wavelets_list = ['db4']

    fig, axs = plt.subplots(
        nrows=2,
        ncols=len(wavelets_list) * 2,
        figsize=(4 * 2, 3 * len(wavelets_list) * 2),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for i, wavelet_name in enumerate(wavelets_list):
        wavelet_basic = pywt.Wavelet(wavelet_name)
        view_wavelet_filters(wavelet_basic)

        base_lo = wavelet_basic.dec_lo.copy()
        new_lo = [
            coef * (1.0 + (0.1 if idx == 0 else 0.0))
            for idx, coef in enumerate(base_lo)
        ]

        dec_lo, dec_hi, rec_lo, rec_hi = orthogonal_filter_bank(new_lo)

        wavelet_custom = Wavelet(wavelet_name + "_tweaked", filter_bank=(dec_lo, dec_hi, rec_lo, rec_hi))
        view_wavelet_filters(wavelet_custom)

        plot_wavelet(axs, i, wavelet_name, wavelet_basic, wavelet_custom)

    plt.tight_layout()
    plt.show()
