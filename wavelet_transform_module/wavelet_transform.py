"""Wavelet transform creation script."""
from copy import deepcopy

import numpy as np
import pywt
from numpy.typing import NDArray


def get_wavelet(wavelet_name: str) -> pywt.Wavelet:
    """Get the wavelet based on the given name."""

    wavelet_original_name = wavelet_name.split('_')[0]
    basic_filters = pywt.Wavelet(wavelet_original_name).filter_bank

    if '_cust' in wavelet_name:
        filter_to_change = 0
        custom_filters = list(deepcopy(basic_filters))
        for i, filter_value in enumerate(basic_filters[filter_to_change]):
            custom_filters[filter_to_change][i] = filter_value * 1.1

        print(
            basic_filters[filter_to_change][0], custom_filters[filter_to_change][0],
            basic_filters[filter_to_change][0] == custom_filters[filter_to_change][0]
        )

        wavelet_custom_filters = pywt.Wavelet(
            wavelet_name,
            filter_bank=tuple(custom_filters)
        )

        # wavelet_custom_filters.family_name = 'Daubechies'
        # wavelet_custom_filters.short_family_name = 'db'
        wavelet_custom_filters.orthogonal = True
        wavelet_custom_filters.biorthogonal = True
        # wavelet_custom_filters.symmetry = False

        return wavelet_custom_filters

    elif '_reb' in wavelet_name:

        custom_wavelet = pywt.Wavelet(
            wavelet_name,
            filter_bank=basic_filters
        )

        # custom_wavelet.family_name = 'Daubechies'
        # custom_wavelet.short_family_name = 'db'
        custom_wavelet.orthogonal = True
        custom_wavelet.biorthogonal = True
        # custom_wavelet.symmetry = False

        return custom_wavelet

    return pywt.Wavelet(wavelet_name)


def discrete_wavelet_transform(signal: NDArray, wavelet_name: str, level: int | None = None) -> tuple[NDArray, list[NDArray]]:
    """Run DWT."""

    wavelet = get_wavelet(wavelet_name)

    if level == 2:
        cA, cD = pywt.dwt(signal, wavelet, level)
        return cA, [cD]
    else:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        cA_n = coeffs[0]
        cDs = coeffs[1:]
        return cA_n, cDs


def inverse_discrete_wavelet_transform(
        cA: NDArray, cDs: list[NDArray], wavelet_name, levels_for_inversion: list[str] | None = None
):
    """Run inverse DWT."""

    wavelet = get_wavelet(wavelet_name)

    # All coefficients
    if levels_for_inversion is None:
        if len(cDs) == 1:
            return pywt.idwt(cA, cDs[0], wavelet)
        else:
            coeffs = [cA] + cDs
            return pywt.waverec(coeffs, wavelet)

    # If include approximation coefficient
    if "A" in levels_for_inversion:
        cA_mod = cA
    else:
        cA_mod = np.zeros_like(cA)

    # Get specific details coefficients
    n_levels = len(cDs)
    cDs_mod = []
    for i, cd in enumerate(cDs):
        lvl = n_levels - i
        name = f"D{lvl}"
        if name in levels_for_inversion:
            cDs_mod.append(cd)
        else:
            cDs_mod.append(np.zeros_like(cd))

    if len(cDs_mod) == 1:
        return pywt.idwt(cA_mod, cDs_mod[0], wavelet)
    else:
        return pywt.waverec([cA_mod] + cDs_mod, wavelet)
