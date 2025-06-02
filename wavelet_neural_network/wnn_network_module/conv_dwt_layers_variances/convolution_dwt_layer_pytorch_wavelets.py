"""Convolution wavelet DWT layer class for Wavelet Neural Network."""
from copy import deepcopy

import numpy as np
import pywt
import torch
from logger import logger
from pytorch_wavelets import DWT1D, IDWT1D
from torch import Tensor
from torch import nn


class WaveletDWTLayer(nn.Module):
    """Create initial parameters based on given discrete wavelet name."""

    def __init__(self, wavelet_name: str = 'db4', layer_number: int = 0,
                 number_coeffs_for_rec: int = 2, filler_value: float = 10.1) -> None:
        super().__init__()
        self.wavelet_name = wavelet_name
        self.wavelet = pywt.Wavelet(self.wavelet_name)
        initial_scaling_filter = deepcopy(self.wavelet.rec_lo)
        self.weights = nn.Parameter(torch.tensor(initial_scaling_filter, dtype=torch.float32), requires_grad=True)
        self.layer_number = layer_number
        self.number_coeffs_for_rec = number_coeffs_for_rec
        self.filler_value = filler_value  # Out from <-10,10> range, to tell that this is only a filler

    @staticmethod
    def torch_qmf(wavelet_filter: Tensor) -> Tensor:
        """Returns the Quadrature Mirror Filter (QMF) of the input tensor.
        Refactored from PyWavelets library:
        https://pywavelets.readthedocs.io/en/latest/ref/other-functions.html#pywt.qmf"""
        if wavelet_filter.dim() != 1:
            raise ValueError("`wavelet_filter` must be a 1-D tensor.")
        qm_filter = torch.flip(wavelet_filter, dims=[0])
        qm_filter[1::2] = -qm_filter[1::2]
        return qm_filter

    def torch_orthogonal_filter_bank(
            self, scaling_filter: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

    def run_dwt(self, signal: Tensor) -> tuple[Tensor, Tensor]:
        """Run DWT using pytorch_wavelets."""
        if signal.dim() == 2:  # (N, L)
            signal = signal.unsqueeze(1)  # → (N, 1, L)
        elif signal.dim() != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got {signal.dim()}D")

        dec_lo, dec_hi, rec_lo, rec_hi = self.torch_orthogonal_filter_bank(self.weights)
        dwt1d = DWT1D(wave=(dec_lo.tolist(), dec_hi.tolist()), mode='symmetric')
        cA, cD_list = dwt1d(signal)
        cD = cD_list[0]

        return cA, cD

    def run_idwt(
            self, cA: Tensor, cD: Tensor, cDs: Tensor, number_coeffs_for_rec: int = 2
    ) -> Tensor:
        """Run IDWT using pytorch_wavelets."""
        dec_lo, dec_hi, rec_lo, rec_hi = self.torch_orthogonal_filter_bank(self.weights)
        idwt1d = IDWT1D(wave=(rec_lo.tolist(), rec_hi.tolist()), mode='symmetric')

        cA_t = torch.zeros_like(cA)
        cDs_t: list[Tensor] = []

        for cD_number in reversed(range(cDs.size(1))):
            cD_with_pad = cDs[:, cD_number:cD_number + 1, :]

            mask = (cD_with_pad != self.filler_value)
            real_len = int(mask[0, 0].sum().item())
            cD_real = cD_with_pad[:, :, :real_len]

            if cD_number in range(number_coeffs_for_rec):
                cDs_t.append(cD_real)
            else:
                cDs_t.append(torch.zeros_like(cD_real))

        rec = idwt1d((cA_t, cDs_t))
        return rec

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Process each batch element by running DWT and IDWT."""
        logger.debug('{module_name} forward {layer_num}'.format(
            module_name=self.__class__.__name__, layer_num=self.layer_number)
        )

        # Update wavelet with new weights as filter
        self.update_wavelet()

        # Run DWT
        cA, cD = self.run_dwt(x1)

        # Add new cD to cDs
        if self.layer_number == 1:
            cDs = cD
        else:
            pad = cD.new_full((cD.size(0), 1, x2.size(-1) - cD.size(-1)), self.filler_value)
            cD_padded = torch.cat([cD, pad], dim=-1)
            cDs = torch.cat([x2, cD_padded], dim=1)

        reconstructions: Tensor
        if self.layer_number == 1:
            reconstructions = x3
        else:
            # Run IDWT
            R = self.run_idwt(cA, cD, cDs)

            # Add new reconstruction to reconstructions
            if self.layer_number == 2:
                reconstructions = R
            else:
                reconstructions = torch.cat([x3, R], dim=1)

        return cA, cDs, reconstructions
