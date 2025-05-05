"""Main script to run expected analysis."""
from numpy._typing import NDArray

from data_getter.from_af_termination_challenge.get_data import get_from_af_termination_challenge
from data_getter.from_mit_bih_arrhythmia_database.get_data import get_from_mit_bih_arrhythmia_database
from plotters.dwt_decomposition_plotter import DWTDecompositionPlotter
from plotters.signals_plotter import SignalsPlotter
from wavelet_transform import discrete_wavelet_transform, inverse_discrete_wavelet_transform


def get_signals_data(
        signal_time: int,
) -> list[tuple[NDArray, str, NDArray, dict[str, int]] | tuple[list[tuple[NDArray, str, NDArray], dict[str, int]]]]:
    """Get specified number of signals with its details information."""
    h_signals_data = get_from_mit_bih_arrhythmia_database(['100', '103'], signal_time)
    a_signals_data = get_from_mit_bih_arrhythmia_database(['108', '113'], signal_time)
    af_signals_data = get_from_af_termination_challenge(['n01', 'n02'], signal_time)
    return h_signals_data + a_signals_data + af_signals_data


def dwt_plotting(wavelet: str, signal_time: int, decomposition_levels: int = 2):
    """Get signal, get wavelet transforms, finally plot them."""

    signals_plotter = DWTDecompositionPlotter()

    signals_data = get_signals_data(signal_time)

    for signal, tag, qrs_peaks, fields in signals_data:
        print(signal.shape, tag, qrs_peaks.shape, fields)

        cA, cDs = discrete_wavelet_transform(signal, wavelet, level=decomposition_levels)

        signals_plotter.add_signal_with_analysis((signal, tag, qrs_peaks, fields['fs'], cA, cDs, wavelet))

    signals_plotter.compute_plotting()


def main():
    """Get signals, create wavelet transforms, plot results."""

    dwt_plotting('sym4', 20, decomposition_levels=6)

    dwt_plotting('db4', 20, decomposition_levels=6)

    dwt_plotting('db4', 20, decomposition_levels=4)

    dwt_plotting('db4', 20, decomposition_levels=6)

    SignalsPlotter.display_plots()


if __name__ == '__main__':
    main()
