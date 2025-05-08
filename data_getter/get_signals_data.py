"""Function to get a set of signals for analysis"""
from numpy._typing import NDArray

from data_getter.from_af_termination_challenge.get_data import get_from_af_termination_challenge
from data_getter.from_mit_bih_arrhythmia_database.get_data import get_from_mit_bih_arrhythmia_database


def get_signals_data(
        signal_time: int, amounts: dict[str, int] = None
) -> list[tuple[NDArray, str, NDArray, dict[str, int]] | tuple[list[tuple[NDArray, str, NDArray], dict[str, int]]]]:
    """Get specified number of signals with its details information."""

    if amounts is None:
        amounts = {'NSR': 2, 'ARR': 2, 'AFT': 2}

    tags_available = dict(NSR=['100', '103'], ARR=['108', '113'], AFT=['n01', 'n02'])

    signals_tags = {
        f'{tags_group}': tags_available[tags_group][:amount]
        for tags_group, amount in amounts.items()
    }

    h_signals_data = get_from_mit_bih_arrhythmia_database(signals_tags['NSR'], signal_time)
    a_signals_data = get_from_mit_bih_arrhythmia_database(signals_tags['ARR'], signal_time)
    af_signals_data = get_from_af_termination_challenge(signals_tags['AFT'], signal_time)
    return h_signals_data + a_signals_data + af_signals_data
