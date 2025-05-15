"""
Get ECG data from physionet called "AF Termination Challenge Database".
Source: https://physionet.org/content/aftdb/1.0.0/
"""
import os

import wfdb
from numpy.typing import NDArray

from data_getter.utils import get_signal_subset, get_qrs_peaks, normalize_signal

DATA_SOURCE = os.path.join(
    'data_getter', 'from_af_termination_challenge', 'af-termination-challenge-database-1.0.0', 'learning-set')
DATA_FREQUENCY = 128  # From documentation


def get_from_af_termination_challenge(
        signal_tags: list[str], seconds: int, get_subsignals: bool = False, normalization: str = 'max-abs',
) -> tuple[NDArray, str, NDArray, dict[str, int]] | tuple[list[tuple[NDArray, str, NDArray], dict[str, int]]]:
    """Main function to get expected signals amount and types with the proper time."""

    signals_list: list[tuple[NDArray, str, NDArray, dict[str, int]]] = []

    for tag in signal_tags:
        record_path = os.path.join(DATA_SOURCE, tag)

        signal, fields = wfdb.rdsamp(record_path)

        if get_subsignals:
            signals = [signal[:, 0], signal[:, 1]] if signal.ndim == 2 else [signal]

        else:
            signals = [signal[:, 0]] if signal.ndim == 2 else [signal]

        for signal in signals:

            try:
                annotation = wfdb.rdann(record_path, 'qrs')
                qrs_locs = annotation.sample
            except FileNotFoundError:
                print("⚠️Could not find qrs file.")
                qrs_locs = []

            signal_subset = get_signal_subset(signal, fields['fs'], seconds)

            signal_subset_normalized = normalize_signal(signal_subset, normalization_mode=normalization)

            qrs_peaks = get_qrs_peaks(signal_subset_normalized, fields['fs'], qrs_locs, seconds)

            signals_list.append((signal_subset_normalized, tag, qrs_peaks, fields))

    return signals_list
