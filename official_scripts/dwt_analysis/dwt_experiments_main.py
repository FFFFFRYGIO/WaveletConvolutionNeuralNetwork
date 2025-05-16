"""Run various experiments using ECGSignal class and dwt_features script."""
from numpy.typing import NDArray

from dwt_features import get_signals_list
from ecg_signal import ECGSignalContent
from official_scripts.dwt_analysis.experiments_plotter import DWTExperimentsPlotter


def main():
    """Run DWT and IDWT experiments."""
    seconds = 5
    signal_amounts = {'ARR': 1, 'CHF': 1, 'NSR': 1}
    normalize_denoise_combinations = [[True, False]]  # [[False, False], [True, False], [True, True], [False, True]]
    wavelets = ['db4', 'sym4']
    decomposition_levels = 3
    reconstruction_combinations_set: list[list[str]] = []

    signals_data, frequency = get_signals_list(seconds, signal_amounts)

    signals_contents_objects_list: list[ECGSignalContent] = []

    for wavelet in wavelets:
        for signal, tag in signals_data:
            for normalize, denoise in normalize_denoise_combinations:
                signal_object = ECGSignalContent(signal, tag, frequency, wavelet, normalize, denoise)
                signal_object.run_dwt(decomposition_levels)
                signal_object.set_reconstruction_combinations(combinations=reconstruction_combinations_set)
                signal_object.run_idwt()
                signals_contents_objects_list.append(signal_object)

    plotter = DWTExperimentsPlotter(signals_contents_objects_list, frequency)
    plotter.compute_plotting()
    plotter.display_plots(maximized=False)


if __name__ == '__main__':
    main()
