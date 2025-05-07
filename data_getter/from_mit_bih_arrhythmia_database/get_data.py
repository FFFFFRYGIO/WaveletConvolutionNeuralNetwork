"""
Get ECG data from physionet called "AF Termination Challenge Database".
Source: https://physionet.org/content/mitdb/1.0.0/
Documentation: https://archive.physionet.org/physiobank/database/html/mitdbdir/records.htm
"""
import os

import wfdb
from numpy.typing import NDArray

from data_getter.utils import get_signal_subset, get_qrs_peaks, normalize_signal

DATA_SOURCE = os.path.join(
    'data_getter', 'from_mit_bih_arrhythmia_database', 'mit-bih-arrhythmia-database-1.0.0')
DATA_FREQUENCY = 360  # From documentation

TAGS_SHIFTS = {
    '100': 11 * 60 + 3,  # Normal sinus rhythm
    '101': 1 * 60 + 34,  # Normal sinus rhythm - is it?
    '103': 1 * 60 + 9,  # Normal sinus rhythm
    '108': 10 * 60 + 55,  # Sinus arrhythmia
    '113': 29 * 60 + 1  # Sinus arrhythmia
}


def get_from_mit_bih_arrhythmia_database(
        signal_tags: list[str], seconds: int, get_subsignals: bool = False
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
            sample_shift_seconds = TAGS_SHIFTS.get(tag, None)

            signal_subset = get_signal_subset(signal, fields['fs'], seconds, sample_shift_seconds=sample_shift_seconds)

            signal_subset_normalized = normalize_signal(signal_subset)

            qrs_peaks = get_qrs_peaks(signal_subset_normalized, fields['fs'])

            signals_list.append((signal_subset_normalized, tag, qrs_peaks, fields))

    return signals_list
