"""Function to get a set of signals for analysis"""
from numpy._typing import NDArray

from data_getter.from_af_termination_challenge.get_data import get_from_af_termination_challenge
from data_getter.from_mit_bih_arrhythmia_database.get_data import get_from_mit_bih_arrhythmia_database


def get_signals_data(
        signal_time: int, amounts: dict[str, int] = None, normalization: str | None = 'peak'
) -> list[tuple[NDArray, str, NDArray, dict[str, int]] | tuple[list[tuple[NDArray, str, NDArray], dict[str, int]]]]:
    """Get specified number of signals with its details information."""

    if amounts is None:
        amounts = {'NSR': 2, 'ARR': 2, 'AFT': 2}

    tags_available = dict(NSR=['100', '103'], ARR=['108', '113'], AFT=['n01', 'n02'])

    signals_tags = {
        f'{tags_group}': tags_available[tags_group][:amount]
        for tags_group, amount in amounts.items()
    }

    h_signals_data = get_from_mit_bih_arrhythmia_database(signals_tags['NSR'], signal_time, normalization=normalization)
    a_signals_data = get_from_mit_bih_arrhythmia_database(signals_tags['ARR'], signal_time, normalization=normalization)
    af_signals_data = get_from_af_termination_challenge(signals_tags['AFT'], signal_time, normalization=normalization)
    return h_signals_data + a_signals_data + af_signals_data


def get_signals_data_from_pywt():
    """Function to get example data from pywt module"""

    import matplotlib
    import numpy as np
    import pywt.data
    from matplotlib import pyplot as plt

    matplotlib.use("TkAgg")

    signal = pywt.data.ecg()

    freq = 1
    duration = len(signal) / freq
    time_signal = np.linspace(0, duration, num=len(signal))
    # qrs_peaks = get_qrs_peaks(signal, freq)

    plt.plot(time_signal, signal)
    plt.title(f"ECG: with QRS (l={len(signal)})")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [mV]")
    plt.xlim(0, duration)
    plt.grid(True)
    # for qrs in qrs_peaks:
    #     plt.axvline(x=qrs / freq, color='r', linestyle='--', alpha=0.1)

    plt.show()
