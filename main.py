"""Main script to run expected analysis"""
from data_getter.from_github.get_data import get_from_github
from data_getter.from_af_termination_challenge.get_data import get_from_af_termination_challenge
from signals_plotter import SignalsPlotter
from wavelet_transform import wavelet_transform


def dwt_plotting(wavelet: str, signal_time: int):
    """Get signal, get wavelet transforms, finally plot them."""

    decomposition_levels = 3

    h_signals_data = get_from_github(2, 'NSR', signal_time)

    for signal, tag, qrs_peaks, fields in h_signals_data:
        print(signal.shape, tag, qrs_peaks.shape, fields)

        cA, cDs = wavelet_transform(signal, wavelet, level=decomposition_levels)

        signals_plotter.add_signal_with_analysis((signal, tag, qrs_peaks, fields['fs'], cA, cDs, wavelet))

    af_signals_data = get_from_af_termination_challenge(['n01'], signal_time)

    for signal, tag, qrs_peaks, fields in af_signals_data:
        print(signal.shape, tag, qrs_peaks.shape, fields)

        cA, cDs = wavelet_transform(signal, wavelet, level=decomposition_levels)

        signals_plotter.add_signal_with_analysis((signal, tag, qrs_peaks, fields['fs'], cA, cDs, wavelet))

    signals_plotter.compute_plotting_signals()



def main():
    """Get signals, create wavelet transforms, plot results"""

    dwt_plotting('db12', 10)

    signals_plotter.display_plots()


if __name__ == '__main__':
    signals_plotter = SignalsPlotter()
    main()
