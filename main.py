"""Main script to run expected analysis"""

from data_getter.from_af_termination_challenge.get_data import get_from_af_termination_challenge
from data_getter.from_mit_bih_arrhythmia_database.get_data import get_from_mit_bih_arrhythmia_database
from signals_plotter import SignalsPlotter
from wavelet_transform import wavelet_transform


def dwt_plotting(wavelet: str, signal_time: int, decomposition_levels: int = 2):
    """Get signal, get wavelet transforms, finally plot them."""

    h_signals_data = get_from_mit_bih_arrhythmia_database(['100', '103', '108', '113'], signal_time)

    for signal, tag, qrs_peaks, fields in h_signals_data:
        print(signal.shape, tag, qrs_peaks.shape, fields)

        cA, cDs = wavelet_transform(signal, wavelet, level=decomposition_levels)

        signals_plotter.add_signal_with_analysis((signal, tag, qrs_peaks, fields['fs'], cA, cDs, wavelet))

    af_signals_data = get_from_af_termination_challenge(['n01', 'n02'], signal_time)

    for signal, tag, qrs_peaks, fields in af_signals_data:
        print(signal.shape, tag, qrs_peaks.shape, fields)

        cA, cDs = wavelet_transform(signal, wavelet, level=decomposition_levels)

        signals_plotter.add_signal_with_analysis((signal, tag, qrs_peaks, fields['fs'], cA, cDs, wavelet))


def main():
    """Get signals, create wavelet transforms, plot results"""

    dwt_plotting('db4', 20, decomposition_levels=2)
    signals_plotter.compute_plotting_signals()
    signals_plotter.signals_set = []

    dwt_plotting('db4', 20, decomposition_levels=4)
    signals_plotter.compute_plotting_signals()
    signals_plotter.signals_set = []

    dwt_plotting('db4', 20, decomposition_levels=6)
    signals_plotter.compute_plotting_signals()

    signals_plotter.display_plots()


if __name__ == '__main__':
    signals_plotter = SignalsPlotter()
    main()
