"""Run various experiments using ECGSignal class and dwt_features script."""
from dwt_features import get_signals_list, run_signals_analysis


def main():
    """Run DWT and IDWT experiments."""
    seconds: int | None = 10
    signal_amounts: dict[str, int] = {'ARR': 1, 'CHF': 1, 'NSR': 1}
    signals_data, frequency = get_signals_list(seconds, signal_amounts)

    wavelets_list: list[str] = ['db4', 'db6', 'sym4']
    decomposition_levels: list[int] = [4]
    denoise_combinations: set[bool] = {False}
    reconstruction_combinations_set: list[str | list[str]] = ['first_two', 'last_two']

    run_signals_analysis(
        signals_data, frequency, wavelets_list, decomposition_levels,
        denoise_combinations, reconstruction_combinations_set,
    )


if __name__ == '__main__':
    main()
