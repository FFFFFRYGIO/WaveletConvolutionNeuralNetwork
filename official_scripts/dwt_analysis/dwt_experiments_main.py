"""Run various experiments using ECGSignal class and dwt_features script."""
from dwt_features import get_signals_list, run_signals_analysis


def main():
    """Run DWT and IDWT experiments."""
    seconds = 5
    signal_amounts = {'ARR': 1, 'CHF': 1, 'NSR': 1}
    signals_data, frequency = get_signals_list(seconds, signal_amounts)

    wavelets_list = ['db4', 'sym4']
    decomposition_levels = 3
    normalize_denoise_combinations = [(True, False)]
    reconstruction_combinations_set: list[list[str]] = []

    run_signals_analysis(
        signals_data, frequency, wavelets_list, decomposition_levels,
        normalize_denoise_combinations, reconstruction_combinations_set,
    )


if __name__ == '__main__':
    main()
