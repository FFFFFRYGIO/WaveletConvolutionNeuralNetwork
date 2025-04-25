"""
Get ECG data from GitHub.
Source: https://github.com/mathworks/physionet_ECG_data/blob/main/ECGData.zip
"""
import os

from numpy.typing import NDArray
from scipy.io import loadmat

from data_getter.utils import get_signal_subset, get_qrs_peaks

DATA_SOURCE = os.path.join('data_getter', 'from_github', 'ECGData', 'ECGData.mat')
DATA_FREQUENCY = 128  # From documentation


def get_from_github(
        amount: str | int, signal_tag: str, seconds: int
) -> tuple[NDArray, str, NDArray, dict[str, int]] | tuple[list[tuple[NDArray, str, NDArray]], dict[str, int]]:
    """Main ECGData function to get expected signals amount and types with the proper time."""

    if (isinstance(amount, str) and amount != 'all') or (isinstance(amount, int) and amount <= 0):
        raise ValueError(f'Bad amount value: {amount=}')

    mat_data = loadmat(DATA_SOURCE)
    raw_data = mat_data['ECGData']
    source_signals, labels = raw_data[0, 0]

    fields = {'fs': DATA_FREQUENCY}

    signals_list: list[tuple[NDArray, str, NDArray]] = []

    match amount:

        case 1:
            for source_signal, label in zip(source_signals, labels):
                if label == signal_tag:
                    signal_subset = get_signal_subset(source_signal, DATA_FREQUENCY, seconds)
                    return signal_subset, signal_tag, get_qrs_peaks(signal_subset), fields

        case 'all':
            for source_signal, label in zip(source_signals, labels):
                if signal_tag == 'all' or label == signal_tag:
                    signal_subset = get_signal_subset(source_signal, DATA_FREQUENCY, seconds)
                    signals_list.append(
                        (signal_subset, label, get_qrs_peaks(signal_subset))
                    )

        case _:
            for i, source_signal, label in enumerate(zip(source_signals, labels)):
                if signal_tag == 'all' or label == signal_tag:
                    signal_subset = get_signal_subset(source_signal, DATA_FREQUENCY, seconds)
                    signals_list.append(
                        (signal_subset, label, get_qrs_peaks(signal_subset))
                    )
                    if i + 1 == amount:
                        break

    return signals_list, fields
