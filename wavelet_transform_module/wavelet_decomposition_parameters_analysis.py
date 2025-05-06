"""Wavelet decomposition filters analysis."""
import matplotlib
import pywt
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


def view_wavelet_filters(wavelet_name: str):
    """View default wavelet filters."""
    print(f'Analysing {wavelet_name=}')

    wavelet = pywt.Wavelet(wavelet_name)

    print("Decomposition low-pass filter (h):", wavelet.dec_lo)
    print("Decomposition high-pass filter (g):", wavelet.dec_hi)
    print("Reconstruction low-pass filter (h̃):", wavelet.rec_lo)
    print("Reconstruction high-pass filter (ĝ):", wavelet.rec_hi)


def plot_wavelet(plot_ax, wavelet_name: str):
    """Plot wavelet scaling and base function."""
    print(f'Plotting {wavelet_name=}')

    wavelet = pywt.Wavelet(wavelet_name)
    phi, psi, x = wavelet.wavefun()

    scaling_plot = plot_ax[0]
    scaling_plot.plot(x, phi)
    scaling_plot.set_title(f"{wavelet_name} Scaling Function φ(x)")
    scaling_plot.set_xlabel("x")
    scaling_plot.set_ylabel("φ(x)")
    scaling_plot.grid(True)

    wavelet_plot = plot_ax[1]
    wavelet_plot.plot(x, psi)
    wavelet_plot.set_title(f"{wavelet_name} Wavelet Function ψ(x)")
    wavelet_plot.set_xlabel("x")
    wavelet_plot.set_ylabel("ψ(x)")
    wavelet_plot.grid(True)


if __name__ == '__main__':
    wavelets_list = ['db4', 'db6', 'db12', 'db16']

    fig, axs = plt.subplots(len(wavelets_list), 2, figsize=(3 * len(wavelets_list), 4 * 2))

    for i, wt in enumerate(wavelets_list):
        view_wavelet_filters(wt)
        plot_wavelet(axs[i], wt)

    plt.tight_layout()
    plt.show()
