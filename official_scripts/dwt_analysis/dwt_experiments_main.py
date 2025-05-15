"""Run various experiments using ECGSignal class and dwt_features script."""
from numpy.typing import NDArray

from dwt_features import get_signals_list
from ecg_signal import ECGSignalContent
from official_scripts.dwt_analysis.experiments_plotter import DWTExperimentsPlotter


def main():
    """Run DWT and IDWT experiments."""
    seconds = 5
    signal_amounts = {'ARR': 1, 'CHF': 1, 'NSR': 1}
    wavelets = ['db4']
    decomposition_levels = 6
    reconstruction_combinations_set: list[list[str]] = []

    signals_data, frequency = get_signals_list(seconds, signal_amounts)

    signals_contents_objects_list: list[ECGSignalContent] = []
    for wavelet in wavelets:
        for signal, tag in signals_data:
            signals_contents_objects_list.append(ECGSignalContent(signal, tag, frequency, wavelet))

    signals_contents_dicts: list[dict[str, tuple[str, NDArray] | list[tuple[str, NDArray]]]] = []
    for signal_content_object in signals_contents_objects_list:
        signal_content_object.run_dwt(decomposition_levels=decomposition_levels)
        signal_content_object.set_reconstruction_combinations(reconstruction_combinations_set)
        signal_content_object.run_idwt()
        signals_contents_dicts.append(signal_content_object.get_signal_content())

    plotter = DWTExperimentsPlotter(signals_contents_dicts, frequency)
    plotter.compute_plotting()
    plotter.display_plots()


if __name__ == '__main__':
    main()
