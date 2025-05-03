"""Main script to run expected analysis"""
import numpy as np
import pywt
from matplotlib import pyplot as plt

from data_getter.from_github.get_data import get_from_github
from data_getter.from_af_termination_challenge.get_data import get_from_af_termination_challenge
from signals_plotter import SignalsPlotter
from wavelet_transform import wavelet_transform


def dwt_plotting(wavelet: str, signal_time: int):
    """Get signal, get wavelet transforms, finally plot them."""

    decomposition_levels = 3

    h_signals_data = get_from_github(2, 'NSR', signal_time)

    for signal, tag, qrs_peaks, fields in h_signals_data:
        print(signal.shape, tag, qrs_peaks.shape, fields)

        cA, cDs = wavelet_transform(signal, wavelet, level=decomposition_levels)

        signals_plotter.add_signal_with_analysis((signal, tag, qrs_peaks, fields['fs'], cA, cDs, wavelet))

    af_signals_data = get_from_af_termination_challenge(['n01'], signal_time)

    for signal, tag, qrs_peaks, fields in af_signals_data:
        print(signal.shape, tag, qrs_peaks.shape, fields)

        cA, cDs = wavelet_transform(signal, wavelet, level=decomposition_levels)

        signals_plotter.add_signal_with_analysis((signal, tag, qrs_peaks, fields['fs'], cA, cDs, wavelet))

    signals_plotter.compute_plotting_signals()


def cwt_plotting(wavelet_name: str, signal_time: int):
    """Plot continuous wavelet transform for 3 signals side by side."""

    h_signals = get_from_github(2, 'NSR', signal_time)
    af_signals = get_from_af_termination_challenge(['n01'], signal_time)
    all_signals = h_signals + af_signals

    cw = pywt.ContinuousWavelet(wavelet_name)

    scales = np.arange(1, 31)

    fig, axs = plt.subplots(
        nrows=2,
        ncols=len(all_signals),
        figsize=(10 * 2, 3 * len(all_signals)),
        sharex=True,
        sharey='row',
    )

    for idx, (signal, tag, qrs_peaks, fields) in enumerate(all_signals):

        sig_ax = axs[0, idx]
        sig_ax.plot(signal)
        sig_ax.set_title(f"ECG: {tag} with QRS (len={len(signal)})")
        sig_ax.set_xlabel("Probes")
        sig_ax.set_ylabel("Amplitude [mV]")
        sig_ax.grid(True)
        for qrs in qrs_peaks:
            sig_ax.axvline(x=qrs, color='r', linestyle='--', alpha=0.1)

        coeffs, freqs = pywt.cwt(signal, scales, cw)

        wt_ax = axs[1, idx]
        wt_ax.imshow(np.abs(coeffs), aspect='auto')
        wt_ax.invert_yaxis()
        wt_ax.set_title(f"Wavelet transform {tag} (len={signal.size})")
        wt_ax.set_xlabel("Probes")
        wt_ax.set_ylabel("Scale")


def main():
    """Get signals, create wavelet transforms, plot results"""

    dwt_plotting('db12', 10)
    cwt_plotting('morl', 5)

    signals_plotter.display_plots()


if __name__ == '__main__':
    signals_plotter = SignalsPlotter()
    main()
