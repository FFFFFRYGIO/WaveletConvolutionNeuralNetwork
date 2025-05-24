import matplotlib
import numpy as np
import pywt
from matplotlib import pyplot as plt

from data_analysis.data_getter.from_af_termination_challenge.get_data import get_from_af_termination_challenge
from data_analysis.data_getter.from_github.get_data import get_from_github


def cwt_plotting(wavelet_name: str, signal_time: int):
    """Plot continuous wavelet transform for 3 signals side by side."""

    h_signals = get_from_github(2, 'NSR', signal_time)
    af_signals = get_from_af_termination_challenge(['n01'], signal_time)
    all_signals = h_signals + af_signals

    cw = pywt.ContinuousWavelet(wavelet_name)

    scales = np.arange(1, 31)

    matplotlib.use("TkAgg")

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
