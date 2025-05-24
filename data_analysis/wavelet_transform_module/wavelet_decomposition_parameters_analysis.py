"""Wavelet decomposition filters analysis."""
import matplotlib
import numpy as np
import pywt
import matplotlib.pyplot as plt
from pywt import Wavelet, orthogonal_filter_bank, qmf

matplotlib.use("TkAgg")


def view_wavelet_filters(wavelet: Wavelet):
    """View default wavelet filters."""
    print(f'Analysing Wavelet: {wavelet}')

    print("Decomposition low-pass filter (h):", wavelet.dec_lo)
    print("Decomposition high-pass filter (g):", wavelet.dec_hi)
    print("Reconstruction low-pass filter (h̃):", wavelet.rec_lo)
    print("Reconstruction high-pass filter (ĝ):", wavelet.rec_hi)


def plot_wavelet(scaling_plot, wavelet_plot, wavelet: Wavelet):
    """Plot wavelet scaling and base function."""
    print(f'Plotting Wavelet: {wavelet}')

    phi, psi, x = wavelet.wavefun()[:3]

    scaling_plot.plot(x, phi)
    scaling_plot.set_title(f"{wavelet.name} Scaling φ(x)")
    scaling_plot.set_xlabel("x")
    scaling_plot.set_ylabel("φ(x)")
    scaling_plot.grid(True)

    wavelet_plot.plot(x, psi)
    wavelet_plot.set_title(f"{wavelet.name} Wavelet ψ(x)")
    wavelet_plot.set_xlabel("x")
    wavelet_plot.set_ylabel("ψ(x)")
    wavelet_plot.grid(True)


if __name__ == '__main__':
    wavelets_list = ['db4', 'db8', 'sym4']

    plot_custom_wavelet = True

    if plot_custom_wavelet:
        fig, axs = plt.subplots(
            nrows=3,
            ncols=len(wavelets_list) * 2,
            figsize=(10 * 2, 2 * len(wavelets_list)),
            squeeze=False,
            sharex=True,
            sharey=True,
        )

        for i, wavelet_name in enumerate(wavelets_list):
            wavelet_basic = pywt.Wavelet(wavelet_name)

            view_wavelet_filters(wavelet_basic)
            plot_wavelet(axs[0, 0 + 2 * i], axs[0, 1 + 2 * i], wavelet_basic)

            base_lo = np.array(wavelet_basic.dec_lo, dtype=float)
            new_lo = base_lo.copy()
            new_lo[0] *= 1.0001
            new_lo *= (np.sqrt(2) / new_lo.sum())

            dec_hi = qmf(new_lo)
            dec_lo, rec_lo, rec_hi = new_lo, new_lo[::-1], dec_hi[::-1]

            wavelet_custom = Wavelet(
                wavelet_name + "_tweaked",
                filter_bank=(dec_lo, dec_hi, rec_lo, rec_hi)
            )

            view_wavelet_filters(wavelet_custom)
            plot_wavelet(axs[1, 0 + 2 * i], axs[1, 1 + 2 * i], wavelet_custom)

            wavelet_custom2 = Wavelet(
                wavelet_name + "_mine",
                filter_bank=wavelet_basic.filter_bank
            )

            view_wavelet_filters(wavelet_custom2)
            plot_wavelet(axs[2, 0 + 2 * i], axs[2, 1 + 2 * i], wavelet_custom2)

            # # Custom wavelet built with orthogonal_filter_bank
            #
            # phi, psi, x = wavelet_basic.wavefun()
            #
            # filters_for_customizing_wavelets = [
            #     wavelet_basic.dec_lo,
            #     # wavelet_basic.dec_hi,  # MALFUNCTION
            #     wavelet_basic.rec_lo,
            #     # wavelet_basic.rec_hi,  # MALFUNCTION
            # ]
            #
            # for filter_num, filter_set in enumerate(filters_for_customizing_wavelets):
            #     filters = pywt.orthogonal_filter_bank(filter_set)
            #
            #     wavelet_custom_temp = pywt.Wavelet(
            #         name="custom_db",
            #         filter_bank=filters
            #     )
            #
            #     view_wavelet_filters(wavelet_custom_temp)
            #     plot_wavelet(axs[filter_num + 2, 0 + 2 * i], axs[filter_num + 2, 1 + 2 * i], wavelet_custom_temp)

    else:
        fig, axs = plt.subplots(
            nrows=len(wavelets_list),
            ncols=2,
            figsize=(2 * len(wavelets_list), 4 * 2),
            squeeze=False,
            sharex=True,
            sharey=True,
        )

        for i, wavelet_name in enumerate(wavelets_list):
            wavelet_basic = pywt.Wavelet(wavelet_name)
            view_wavelet_filters(wavelet_basic)

            plot_wavelet(axs[i, 0], axs[i, 1], wavelet_basic)

    plt.tight_layout()
    plt.show()
