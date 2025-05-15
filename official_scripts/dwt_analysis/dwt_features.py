"""Here are implemented features for running multiple DWT decompositions and IDWT."""
import os

from numpy.typing import NDArray
from scipy.io import loadmat


def get_signals_list(signal_time: int, amounts: dict[str, int] = None) -> tuple[list[tuple[NDArray, str]], int]:
    """Get a list of signals specified in the main."""

    data_source = os.getenv('DATA_SOURCE')
    mat_data = loadmat(data_source)
    raw_data = mat_data['ECGData']
    source_signals, labels = raw_data[0, 0]

    signal_classes_first_indexes = {'ARR': 0, 'CHF': 96, 'NSR': 96 + 30}

    if amounts is None:  # Max amounts
        amounts = {'ARR': 96, 'CHF': 30, 'NSR': 36}

    # Get signal frequency based on documentation
    frequency = int(os.getenv('ECGDATA_FREQUENCY'))
    duration_samples = int(signal_time * frequency)

    signals: list[tuple[NDArray, str]] = []

    for signal_class, amount in amounts.items():
        first_index = signal_classes_first_indexes[signal_class]
        for signal_num in range(first_index, first_index + amount):
            signal = source_signals[signal_num][:duration_samples]
            signals.append((signal, signal_class))

    return signals, frequency
