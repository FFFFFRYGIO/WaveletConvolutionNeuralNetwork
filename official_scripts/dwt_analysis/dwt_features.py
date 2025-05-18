"""Here are implemented features for running multiple DWT decompositions and IDWT."""
import os

from numpy.typing import NDArray
from scipy.io import loadmat

from ecg_signal import ECGSignalContent
from experiments_plotter import DWTExperimentsPlotter


def get_signals_list(
        signal_time: int | None = None, amounts: dict[str, int] = None
) -> tuple[list[tuple[NDArray, str]], int]:
    """Get a list of signals specified in the main script."""

    data_source = os.getenv('DATA_SOURCE')
    mat_data = loadmat(data_source)
    raw_data = mat_data['ECGData']
    source_signals, labels = raw_data[0, 0]

    signal_classes_first_indexes = {'ARR': 0, 'CHF': 96, 'NSR': 96 + 30}

    if amounts is None:  # Max amounts
        amounts = {'ARR': 96, 'CHF': 30, 'NSR': 36}

    # Get signal frequency based on documentation
    frequency = int(os.getenv('ECGDATA_FREQUENCY'))
    duration_samples = len(source_signals[0])
    if signal_time is not None:
        duration_samples = int(signal_time * frequency)

    signals: list[tuple[NDArray, str]] = []

    for signal_class, amount in amounts.items():
        first_index = signal_classes_first_indexes[signal_class]
        for signal_num in range(first_index, first_index + amount):
            signal = source_signals[signal_num][:duration_samples]
            signals.append((signal, signal_class))

    return signals, frequency


def run_signals_analysis(
        signals_data: list[tuple[NDArray, str]],
        frequency: int,
        wavelets_list: list[str],
        decomposition_levels: list[int],
        denoise_combinations: set[bool],
        reconstruction_combinations_set: list[str | list[str]] | None = None
) -> None:
    """Run ecg signals analysis for configuration specified in the main script."""

    if reconstruction_combinations_set is None:
        reconstruction_combinations_set = []
    signals_contents_objects_list: list[ECGSignalContent] = []

    for wavelet in wavelets_list:
        for decomposition_level in decomposition_levels:
            for signal, tag in signals_data:
                for if_denoise in denoise_combinations:
                    signal_object = ECGSignalContent(signal, tag, frequency, wavelet, if_denoise)
                    signal_object.run_dwt(decomposition_level)
                    signal_object.set_reconstruction_combinations(combinations=reconstruction_combinations_set)
                    signal_object.run_idwt()
                    signals_contents_objects_list.append(signal_object)

    plotter = DWTExperimentsPlotter(signals_contents_objects_list, frequency)
    plotter.compute_plotting()
    plotter.display_plots(maximized=False)
