"""Convolution wavelet DWT layer class for Wavelet Neural Network."""
from copy import deepcopy

import numpy as np
import pywt
import torch
from torch import nn

from logger import logger


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

    @staticmethod
    def torch_qmf(wavelet_filter: torch.Tensor) -> torch.Tensor:
        """Returns the Quadrature Mirror Filter (QMF) of the input tensor.
        Refactored from PyWavelets library:
        https://pywavelets.readthedocs.io/en/latest/ref/other-functions.html#pywt.qmf"""
        if wavelet_filter.dim() != 1:
            raise ValueError("`wavelet_filter` must be a 1-D tensor.")
        qm_filter = torch.flip(wavelet_filter, dims=[0])
        qm_filter[1::2] = -qm_filter[1::2]
        return qm_filter

    def torch_orthogonal_filter_bank(
            self, scaling_filter: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the orthogonal filter bank for the given scaling filter tensor.
        Refactored from PyWavelets library:
        https://pywavelets.readthedocs.io/en/latest/ref/other-functions.html#pywt.orthogonal_filter_bank"""
        if scaling_filter.dim() != 1:
            raise ValueError("`scaling_filter` must be a 1-D tensor.")
        length = scaling_filter.size(0)
        if length % 2 != 0:
            raise ValueError("`scaling_filter` length has to be even.")

        scaling_filter = scaling_filter.to(torch.float64)

        rec_lo = torch.sqrt(torch.tensor(2.0, dtype=scaling_filter.dtype)) * scaling_filter / scaling_filter.sum()
        dec_lo = torch.flip(rec_lo, dims=[0])

        rec_hi = self.torch_qmf(rec_lo)
        dec_hi = torch.flip(rec_hi, dims=[0])

        return dec_lo, dec_hi, rec_lo, rec_hi

    def update_wavelet(self) -> None:
        """Update wavelet with new weights as filter."""
        new_filter_bank = self.torch_orthogonal_filter_bank(self.weights)
        new_filter_bank = [f.tolist() for f in new_filter_bank]
        new_wavelet = pywt.Wavelet(f'cust_{self.wavelet_name}', filter_bank=new_filter_bank)
        new_wavelet.orthogonal = True
        new_wavelet.biorthogonal = True
        self.wavelet = new_wavelet

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
            coeff = coeff_with_pad[np.abs(coeff_with_pad - self.filler_value) > 1e-4].reshape(1, -1)
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

        # Update wavelet with new weights as filter
        self.update_wavelet()

        # Run DWT
        cA, cD = self.run_dwt(signal)

        # Add cD to cDs
        if self.layer_number == 1:
            cDs = np.array([cD])
        else:
            cD_padded = np.pad(cD, ((0, 0), (0, cDs[0].size - cD.size)), constant_values=self.filler_value)
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
        logger.debug('{module_name} forward {layer_num}'.format(
            module_name=self.__class__.__name__, layer_num=self.layer_number)
        )
        signal_tensor = x1
        coeffs_tensor = x2
        reconstructions_tensor = x3

        # Process each element in the batch individually
        cA_batch, cD_batch, R_batch = [], [], []

        for i in range(signal_tensor.shape[0]):
            signal_detached = signal_tensor[i, :].cpu().detach().numpy()
            cDs_detached = coeffs_tensor[i, :].cpu().detach().numpy()
            reconstructions_detached = reconstructions_tensor[i, :].cpu().detach().numpy()

            signal = signal_detached
            cDs = [cD.reshape(1, -1) for cD in cDs_detached]
            reconstructions = [rec.reshape(1, -1) for rec in reconstructions_detached]

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
