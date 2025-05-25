"""Convolution wavelet DWT layer class for Wavelet Neural Network."""
from copy import deepcopy

import numpy as np
import pywt
import torch
from torch import nn


class WaveletDWTLayer(nn.Module):
    """Create initial parameters based on given discrete wavelet name."""

    def __init__(self, wavelet_name: str = 'db4', layer_number: int = 0, filler_value: float = 10.1) -> None:
        super().__init__()
        self.wavelet_name = wavelet_name
        self.wavelet = pywt.Wavelet(self.wavelet_name)
        initial_scaling_filter = deepcopy(self.wavelet.rec_lo)
        self.weights = nn.Parameter(torch.tensor(initial_scaling_filter, dtype=torch.float32), requires_grad=True)
        self.layer_number = layer_number
        self.filler_value = filler_value  # Out from <-10,10> range, to tell that this is only a filler

    def run_dwt(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run DWT on input signal."""
        cA, cD = pywt.dwt(signal, self.wavelet)
        return cA, cD

    def run_idwt(
            self, cA: np.ndarray, cD: np.ndarray, cDs: list[np.ndarray], number_coeffs_for_rec: int = 2
    ) -> np.ndarray:
        """Run inverse DWT based on a coefficient list."""
        waverec_coeffs = [np.zeros_like(cA)]

        for coeff_number, coeff_with_pad in enumerate(cDs[::-1]):
            coeff_bad_shape = coeff_with_pad[np.round(coeff_with_pad, 4) != self.filler_value]
            coeff = np.expand_dims(np.asarray(coeff_bad_shape), axis=0)
            if coeff_number in range(number_coeffs_for_rec):
                waverec_coeffs.append(coeff)
            else:
                waverec_coeffs.append(np.zeros_like(coeff))

        R = pywt.waverec(waverec_coeffs, self.wavelet)
        return R

    def run_wavelet_computation(
            self, signal: np.ndarray, cDs: list[np.ndarray], reconstructions: list[np.ndarray]
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Run DWT and IDWT and return result for a batch element."""

        # Run DWT
        cA, cD = self.run_dwt(signal)

        # Add cD to cDs
        if self.layer_number == 1:
            cDs = np.array([cD])
        else:
            cD_pad_amount = cDs[0].size - cD.size
            cD_padded = np.pad(cD, ((0, 0), (0, cD_pad_amount)), constant_values=self.filler_value)
            cDs.append(cD_padded)

        # Run IDWT and add new reconstruction
        if not self.layer_number == 1:
            R = self.run_idwt(cA, cD, cDs)
            if self.layer_number == 2:
                reconstructions = np.array([R])
            else:
                reconstructions.append(R)

        return cA, cDs, reconstructions

    def forward(
            self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process each batch element by running DWT and IDWT."""
        signal_tensor = x1
        coeffs_tensor = x2
        reconstructions_tensor = x3

        # Process each element in the batch individually
        cA_batch, cD_batch, R_batch = [], [], []

        for i in range(signal_tensor.shape[0]):
            signal_listed = signal_tensor[i, :].tolist()
            cDs_listed = coeffs_tensor[i, :].tolist()
            reconstructions_listed = reconstructions_tensor[i, :].tolist()

            if self.layer_number == 1:
                signal = np.expand_dims(np.asarray(signal_listed[0]), axis=0)
            else:
                signal = np.asarray(signal_listed)
            cDs = [np.asarray(cD) for cD in cDs_listed]
            reconstructions = [np.asarray(rec) for rec in reconstructions_listed]

            new_cA, new_cDs, new_rec = self.run_wavelet_computation(signal, cDs, reconstructions)

            # Add element to batch segment
            cA_batch.append(new_cA)
            cD_batch.append(new_cDs)
            R_batch.append(new_rec)

        # Stack the results back into batches
        cA_np = np.stack(cA_batch)
        cD_np = np.stack(cD_batch)
        R_np = np.stack(R_batch)

        # Convert NumPy arrays back to PyTorch tensors and require gradients
        x1_r = torch.tensor(cA_np, dtype=x1.dtype, device=x1.device, requires_grad=True)
        x2_r = torch.tensor(cD_np, dtype=x2.dtype, device=x2.device, requires_grad=True)
        x3_r = torch.tensor(R_np, dtype=x3.dtype, device=x3.device, requires_grad=True)

        return x1_r, x2_r, x3_r
