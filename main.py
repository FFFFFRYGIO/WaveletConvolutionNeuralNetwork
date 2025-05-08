"""Main script to run expected analysis."""
from data_getter.get_signals_data import get_signals_data
from plotters.dwt_decomposition_plotter import DWTDecompositionPlotter
from plotters.inverse_dwt_plotter import InverseDWTPlotter
from plotters.signals_plotter import SignalsPlotter
from wavelet_transform_module.wavelet_transform import discrete_wavelet_transform, inverse_discrete_wavelet_transform


def dwt_plotting(wavelet: str, signal_time: int, decomposition_levels: int = 2):
    """Get signal, get wavelet transforms, finally plot them."""

    signals_plotter = DWTDecompositionPlotter()

    signals_data = get_signals_data(signal_time)

    for signal, tag, qrs_peaks, fields in signals_data:
        print(f"{signal.shape=}, {tag=}, {qrs_peaks.shape=}, {fields['fs']=}")

        cA, cDs = discrete_wavelet_transform(signal, wavelet, level=decomposition_levels)

        signals_plotter.add_signal_with_analysis((signal, tag, qrs_peaks, fields['fs'], cA, cDs, wavelet))

    signals_plotter.compute_plotting()


def inverse_dwt_plotting(wavelet: str, signal_time: int, decomposition_levels: int = 2):
    """Get signals, get wavelet transforms, finally plot different inverse dwt for them."""

    signals_plotter = InverseDWTPlotter()

    signals_data = get_signals_data(signal_time)

    inverse_dwt_levels_operations = [['A', f'D{decomposition_levels}'], ['D2', 'D1']]

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

    signals_plotter.compute_plotting(add_decompositions=True)


def main():
    """Get signals, create wavelet transforms, plot results."""

    run_inverse_dwt = True

    if run_inverse_dwt:
        inverse_dwt_plotting('db4', 10, decomposition_levels=6)

    else:
        dwt_plotting('sym4', 20, decomposition_levels=6)
        dwt_plotting('db4', 20, decomposition_levels=6)
        dwt_plotting('db4', 20, decomposition_levels=4)
        dwt_plotting('db4', 20, decomposition_levels=6)

    SignalsPlotter.display_plots()


if __name__ == '__main__':
    main()
