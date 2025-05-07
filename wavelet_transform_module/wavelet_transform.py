"""Wavelet transform creation script."""
import numpy as np
import pywt
from numpy.typing import NDArray


def discrete_wavelet_transform(signal: NDArray, wavelet, level: int | None = None) -> tuple[NDArray, list[NDArray]]:
    """Run DWT."""

    if level == 2:
        cA, cD = pywt.dwt(signal, wavelet, level)
        return cA, [cD]
    else:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        cA_n = coeffs[0]
        cDs = coeffs[1:]
        return cA_n, cDs


def inverse_discrete_wavelet_transform(
        cA: NDArray, cDs: list[NDArray], wavelet, levels_for_inversion: list[str] | None = None
):
    """Run inverse DWT."""

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
