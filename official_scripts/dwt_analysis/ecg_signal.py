"""Implementation of class for ecg signal computation."""
import numpy as np
from numpy.typing import NDArray
from pywt import Wavelet, dwt, wavedec, waverec
from scipy.signal import medfilt


class ECGSignalContent:
    """Class that stored ECG signal content with specified wavelet and implements methods to manipulate its data."""

    def __init__(self, signal: NDArray, tag: str, fs: int, wavelet: str | Wavelet,
                 normalize_signal: bool, denoise_signal: bool) -> None:
        self.signal = signal
        self.tag = tag
        self.fs = fs

        if denoise_signal:
            self.denoise_signal()
        if normalize_signal:
            self.normalize_signal_max_abs()

        if isinstance(wavelet, str):
            self.wavelet = Wavelet(wavelet)
        elif isinstance(wavelet, Wavelet):
            self.wavelet = wavelet
        else:
            raise TypeError('Wavelet must be str or pywt.Wavelet')

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
        if decomposition_levels == 2:
            cA, cD = dwt(self.signal, self.wavelet, 2)
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
            self, combinations: list[list[str]] | None = None, add_first_two=True, add_last_two=True, add_full_reconstruction=False
    ) -> None:
        """Add IDWT reconstruction combinations for calculations."""
        if len(combinations):
            self.reconstruction_combinations.extend(combinations)
        if add_first_two:
            self.reconstruction_combinations.append(['D1', 'D2'])
        if add_last_two:
            self.reconstruction_combinations.append(['A', f'D{len(self.dwt_decomposition) - 1}'])
        if add_full_reconstruction:
            self.reconstruction_combinations.append('full_reconstruction')

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
