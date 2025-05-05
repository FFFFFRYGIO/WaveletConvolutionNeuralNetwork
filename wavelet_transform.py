"""Wavelet transform creation script."""
import numpy as np
import pywt
from numpy.typing import NDArray


def discrete_wavelet_transform(signal: NDArray, wavelet, level: int = 2) -> tuple[NDArray, list[NDArray]]:
    """Run DWT."""

    if level == 2:
        cA, cD = pywt.dwt(signal, wavelet, level)
        return cA, [cD]
    else:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        cA_n = coeffs[0]
        cDs = coeffs[1:]
        return cA_n, cDs
