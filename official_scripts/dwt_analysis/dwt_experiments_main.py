"""Run various experiments using ECGSignal class and dwt_features script."""
from dwt_features import get_signals_list, run_signals_analysis


def main():
    """Run DWT and IDWT experiments."""
    seconds: int | None = None
    signal_amounts: dict[str, int] = {'ARR': 0, 'CHF': 0, 'NSR': 0}
    signals_data, frequency = get_signals_list(seconds, signal_amounts)

    wavelets_list: list[str] = []
    decomposition_levels: list[int] = []
    normalize_denoise_combinations: list[tuple[bool, bool]] = []
    reconstruction_combinations_set: list[str | list[str]] = []

    run_signals_analysis(
        signals_data, frequency, wavelets_list, decomposition_levels,
        normalize_denoise_combinations, reconstruction_combinations_set,
    )


if __name__ == '__main__':
    main()
