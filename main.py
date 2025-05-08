"""Main script to run expected analysis."""
import numpy as np

from data_getter.get_signals_data import get_signals_data
from plotters.dwt_decomposition_plotter import DWTDecompositionPlotter
from plotters.inverse_dwt_plotter import InverseDWTPlotter
from plotters.signal_statistical_analysis_plotter import SignalStatisticalAnalysisPlotter
from plotters.signals_plotter import SignalsPlotter
from signal_statistical_analysis import data_distribution
from wavelet_transform_module.wavelet_transform import discrete_wavelet_transform, inverse_discrete_wavelet_transform


def dwt_plotting(wavelet: str, signal_time: int, decomposition_levels: int = 2):
    """Get signal, get wavelet transforms, finally plot them."""

    signals_plotter = DWTDecompositionPlotter()

    signals_data = get_signals_data(signal_time)

    for signal, tag, qrs_peaks, fields in signals_data:
        print(f"{signal.shape=}, {tag=}, {qrs_peaks.shape=}, {fields['fs']=}")

        cA, cDs = discrete_wavelet_transform(signal, wavelet, level=decomposition_levels)

        signals_plotter.add_signal_with_analysis((signal, tag, qrs_peaks, fields['fs'], cA, cDs, wavelet))

    signals_plotter.compute_plotting(plot_wavelets=False)


def inverse_dwt_plotting(wavelets_list: list[str], signal_time: int, decomposition_levels: int = 2):
    """Get signals, get wavelet transforms, finally plot different inverse dwt for them."""

    signals_plotter = InverseDWTPlotter()

    signals_data = get_signals_data(signal_time)

    inverse_dwt_levels_operations = [
        ['A', f'D{decomposition_levels}'],
        # [f'D{decomposition_levels}', f'D{decomposition_levels - 1}'],
        ['D2', 'D1'],
    ]

    for wavelet in wavelets_list:

        for signal, tag, qrs_peaks, fields in signals_data:
            print(f"{signal.shape=}, {tag=}, {qrs_peaks.shape=}, {fields['fs']=}")

            cA, cDs = discrete_wavelet_transform(signal, wavelet, level=decomposition_levels)

            inverse_dwt_list = []

            for levels_for_inversion in inverse_dwt_levels_operations:
                inverse_dwt = inverse_discrete_wavelet_transform(cA, cDs, wavelet, levels_for_inversion)
                inverse_dwt_list.append((levels_for_inversion, inverse_dwt))

            signals_plotter.add_signal_with_analysis(
                (signal, tag, qrs_peaks, fields['fs'], cA, cDs, wavelet, inverse_dwt_list)
            )

    signals_plotter.compute_plotting(add_decompositions=True, plot_wavelets=False)


def statistical_analysis(signal_time: int):
    """Get signals, get statistical analysis, finally plot all."""

    signals_plotter = SignalStatisticalAnalysisPlotter()

    signals_data = get_signals_data(signal_time, normalization=None) + get_signals_data(signal_time, normalization='peak')

    for signal, tag, qrs_peaks, fields in signals_data:
        dist = data_distribution(
            signal,
            bins=100,
            density=True,
            compute_kde=True,
            kde_points=500
        )

        hist_counts = dist['histogram']
        bin_edges = dist['bin_edges']

        print("Mean:", dist['mean'])
        print("Median:", dist['median'])
        print("5th percentile:", dist['percentiles'][5])

        xs = dist['kde']['xs']
        kde_density = dist['kde']['density']

        signals_plotter.add_signal_with_analysis(signal, tag, qrs_peaks, fields['fs'], hist_counts)

    signals_plotter.compute_plotting()


def main():
    """Get signals, create wavelet transforms, plot results."""

    analysis_mode = 'stats'
    signal_time = 10

    match analysis_mode:
        case 'inverse_dwt':
            inverse_dwt_plotting(['db4', 'sym4'], signal_time, decomposition_levels=6)

        case 'dwt':
            dwt_plotting('db4', signal_time, decomposition_levels=6)

        case 'stats':
            statistical_analysis(signal_time)

    SignalsPlotter.display_plots(maximized=False)


if __name__ == '__main__':
    main()
