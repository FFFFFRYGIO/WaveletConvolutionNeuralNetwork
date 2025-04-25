"""Wavelet transform creation script"""
import pywt
from numpy.typing import NDArray


def wavelet_transform(signal: NDArray, wavelet) -> tuple[NDArray, NDArray]:
    """Run wavelet transform"""

    if signal.ndim == 2:
        signal = signal[:, 0]

    cA, cD = pywt.dwt(signal, wavelet)

    return cA, cD
