"""Implementation of class for ecg signal computation."""
import numpy as np
from numpy.typing import NDArray
from pywt import Wavelet, dwt, wavedec, waverec
from scipy.signal import medfilt


class ECGSignalContent:
    """Class that stored ECG signal content with specified wavelet and implements methods to manipulate its data."""

    def __init__(self, signal: NDArray, tag: str, fs: int, wavelet: str | Wavelet,
                 denoise_signal: bool, normalize_signal: bool = True) -> None:
        self.signal = signal
        self.tag = tag
        self.fs = fs

        if isinstance(wavelet, str):
            self.wavelet = Wavelet(wavelet)
        elif isinstance(wavelet, Wavelet):
            self.wavelet = wavelet
        else:
            raise TypeError('Wavelet must be str or pywt.Wavelet')

        self.mode = ''

        if denoise_signal:
            self.denoise_signal()
            self.mode += 'Den'
        if normalize_signal:
            self.normalize_signal_max_abs()
            self.mode += 'Nor'

        self.mode = self.mode if self.mode else 'Raw'

        self.dwt_decomposition: dict[str, NDArray] = {}
        self.reconstruction_combinations: list[list[str] | str] = []
        self.dwt_reconstructions: list[tuple[str] | str, NDArray] = []

    def normalize_signal_max_abs(self) -> None:
        """Normalize ECG signal with max-abs (from -1 to 1) normalization."""
        max_peak = np.max(np.abs(self.signal))
        if max_peak:
            self.signal = self.signal / max_peak

    def denoise_signal(self) -> None:
        """Remove ECG baseline wander."""
        kernel_size = int(1.0 * self.fs) | 1
        baseline = medfilt(self.signal, kernel_size=kernel_size)
        self.signal = self.signal - baseline

    def run_dwt(self, decomposition_levels: int = 2) -> None:
        """Run DWT for specified decomposition levels."""
        if decomposition_levels == 1:
            cA, cD = dwt(self.signal, self.wavelet)
            self.dwt_decomposition['A'] = cA
            self.dwt_decomposition['D1'] = cD
        else:
            coeffs = wavedec(self.signal, self.wavelet, level=decomposition_levels)
            cA_n = coeffs[0]
            cDs = coeffs[1:]
            self.dwt_decomposition['A'] = cA_n
            for i, cD in enumerate(cDs):
                self.dwt_decomposition[f'D{decomposition_levels - i}'] = cD

    def set_reconstruction_combinations(
            self, combinations: list[list[str]] | None = None
    ) -> None:
        """Add IDWT reconstruction combinations for calculations."""
        for combination in combinations:
            match combination:
                case 'first_two':
                    self.reconstruction_combinations.append(list(self.dwt_decomposition.keys())[:2])
                case 'second_two':
                    self.reconstruction_combinations.append(list(self.dwt_decomposition.keys())[2:4])
                case 'last_two':
                    self.reconstruction_combinations.append(list(self.dwt_decomposition.keys())[-2:])
                case 'all':
                    self.reconstruction_combinations.append(list(self.dwt_decomposition.keys()))
                case _:
                    if isinstance(combination, list):
                        self.reconstruction_combinations.append(combination)
                    else:
                        raise ValueError(f'Bad combination given: {combination}')

    def run_idwt(self) -> None:
        """Run IDWT for specified reconstruction combinations."""
        if not len(self.reconstruction_combinations):
            raise ValueError("No reconstruction combinations set.")
        for reconstruction_combination in self.reconstruction_combinations:
            included_coeffs = []
            for level_name, coeff in list(self.dwt_decomposition.items()):
                if level_name in reconstruction_combination:
                    included_coeffs.append(self.dwt_decomposition[level_name])
                else:
                    included_coeffs.append(np.zeros_like(self.dwt_decomposition[level_name]))
            reconstruction = waverec(included_coeffs, self.wavelet)
            self.dwt_reconstructions.append((reconstruction_combination, reconstruction))
