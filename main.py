"""Main script to run expected analysis."""
import numpy as np

from data_getter.get_signals_data import get_signals_data
from plotters.dwt_decomposition_plotter import DWTDecompositionPlotter
from plotters.inverse_dwt_plotter import InverseDWTPlotter
from plotters.signal_statistical_analysis_plotter import SignalStatisticalAnalysisPlotter
from plotters.signals_plotter import SignalsPlotter
from signal_statistical_analysis import data_distribution, detect_distribution_type
from wavelet_transform_module.wavelet_transform import discrete_wavelet_transform, inverse_discrete_wavelet_transform


def dwt_plotting(wavelet: str, signal_time: int, decomposition_levels: int | None = 2):
    """Get signal, get wavelet transforms, finally plot them."""

    signals_plotter = DWTDecompositionPlotter()

    signals_data = get_signals_data(signal_time)

    for signal, tag, qrs_peaks, fields in signals_data:
        print(f"{signal.shape=}, {tag=}, {qrs_peaks.shape=}, {fields['fs']=}")

        cA, cDs = discrete_wavelet_transform(signal, wavelet, level=decomposition_levels)

        signals_plotter.add_signal_with_analysis((signal, tag, qrs_peaks, fields['fs'], cA, cDs, wavelet))

    signals_plotter.compute_plotting(plot_wavelets=False)


def inverse_dwt_plotting(wavelets_list: list[str], signal_time: int | None, decomposition_levels: int | None = 2):
    """Get signals, get wavelet transforms, finally plot different inverse dwt for them."""

    signals_plotter = InverseDWTPlotter()

    signals_data = get_signals_data(signal_time, amounts={'NSR': 1, 'ARR': 1, 'AFT': 1})

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

    signals_data = (get_signals_data(signal_time, normalization=None) +
                    get_signals_data(signal_time, normalization='max-abs'))

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

        # print("Mean:", dist['mean'])
        # print("Median:", dist['median'])
        # print("5th percentile:", dist['percentiles'][5])

        xs = dist['kde']['xs']
        kde_density = dist['kde']['density']

        detected_distribution = detect_distribution_type(signal, bins=100, use_ks=True)
        print(detected_distribution)

        signals_plotter.add_signal_with_analysis(signal, tag, qrs_peaks, fields['fs'], hist_counts, detected_distribution)

    signals_plotter.compute_plotting()


def main():
    """Get signals, create wavelet transforms, plot results."""

    analysis_mode = 'dwt'
    signal_time = 30
    decomposition_levels = None

    match analysis_mode:
        case 'inverse_dwt':
            inverse_dwt_plotting(['db4', 'db4_reb', 'db4_cust'], signal_time, decomposition_levels=decomposition_levels)

        case 'dwt':
            dwt_plotting('db4', signal_time, decomposition_levels=decomposition_levels)

        case 'stats':
            statistical_analysis(signal_time)

    SignalsPlotter.display_plots(maximized=True)


if __name__ == '__main__':
    # main()

    import pywt
    from copy import deepcopy

    print()

    wavelet = pywt.Wavelet('db4')
    orig_filter = wavelet.rec_lo
    print(f'{orig_filter=}')
    orig_filter_bank = wavelet.filter_bank
    print(f'{orig_filter_bank=}')
    initial_filter = deepcopy(wavelet.rec_lo)
    print(f'{initial_filter=}')
    print()

    params = initial_filter
    print(f'{params=}')
    new_params = [p * (1 + i / 1000) for i, p in enumerate(params)]
    print(f'{new_params=}')
    print()

    filter_back = new_params
    print(f'{orig_filter=}')
    print(f'{filter_back=}')
    print()

    filter_bank1 = pywt.orthogonal_filter_bank(orig_filter)
    filter_bank2 = pywt.orthogonal_filter_bank(filter_back)
    print(f'{filter_bank1=}')
    print(f'{filter_bank2=}')
    print()

    wavelet1 = pywt.Wavelet('cust1', filter_bank=filter_bank1)
    wavelet1.orthogonal = True
    wavelet1.biorthogonal = True
    wavelet2 = pywt.Wavelet('cust2', filter_bank=filter_bank2)
    wavelet2.orthogonal = True
    wavelet2.biorthogonal = True
    print(f'{wavelet1.rec_lo=}')
    print(f'{wavelet2.rec_lo=}')
    print()

    import pandas as pd

    # build a list of dicts
    rows = []
    for wf, wf1, wf2 in zip(wavelet.dec_lo, wavelet1.dec_lo, wavelet2.dec_lo):
        rows.append({
            "wf": wf,
            "wf1": wf1,
            "wf2": wf2,
            "Diff1": wf - wf1,
            "Diff2": wf - wf2,
        })

    # make DataFrame and print
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
