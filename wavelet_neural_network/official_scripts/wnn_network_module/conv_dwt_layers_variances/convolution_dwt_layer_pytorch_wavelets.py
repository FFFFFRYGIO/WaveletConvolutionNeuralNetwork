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

    def __init__(self, wavelet_name: str = 'db4', layer_number: int = 0, filler_value: float = 10.1) -> None:
        super().__init__()
        self.wavelet_name = wavelet_name
        self.wavelet = pywt.Wavelet(self.wavelet_name)
        initial_scaling_filter = deepcopy(self.wavelet.rec_lo)
        self.weights = nn.Parameter(torch.tensor(initial_scaling_filter, dtype=torch.float32), requires_grad=True)
        self.layer_number = layer_number
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

        for cD_number, cD_with_pad in enumerate(cDs.unbind()):
            non_filler_mask = torch.ne(cD_with_pad, self.filler_value)
            cD = cD_with_pad.masked_select(non_filler_mask).view(1, 1, -1)
            if cD_number in range(number_coeffs_for_rec):
                cDs_t.append(cD)
            else:
                cDs_t.append(torch.zeros_like(cD))

        rec = idwt1d((cA_t, cDs_t))
        return rec

    def run_wavelet_computation(
            self, signal: Tensor, cDs: Tensor, reconstructions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run DWT and IDWT and return result for a batch element."""

        # Update wavelet with new weights as filter
        self.update_wavelet()

        # Run DWT
        cA, cD = self.run_dwt(signal)

        # Add cD to cDs
        if self.layer_number == 1:
            cDs = cD
        else:
            pad = cD.new_full((*cD.shape[:-1], cDs.shape[-1] - cD.shape[-1]), self.filler_value)
            cD_padded = torch.cat([cD, pad], dim=-1)
            cDs = torch.cat([cDs, cD_padded])

        # Run IDWT and add new reconstruction
        if not self.layer_number == 1:
            R = self.run_idwt(cA, cD, cDs)
            if self.layer_number == 2:
                reconstructions = R
            else:
                reconstructions = torch.cat([reconstructions, R])

        return cA, cDs, reconstructions

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Process each batch element by running DWT and IDWT."""
        logger.debug('{module_name} forward {layer_num}'.format(
            module_name=self.__class__.__name__, layer_num=self.layer_number)
        )
        signal = x1.unbind()
        cDs = x2.unbind()
        reconstructions = x3.unbind()

        out_cA_list = []
        out_cDs_list = []
        out_recs_list = []

        for signal, cDs, recs in zip(signal, cDs, reconstructions):
            cA_i, cDs_i, rec_i = self.run_wavelet_computation(signal, cDs, recs)
            out_cA_list.append(cA_i)
            out_cDs_list.append(cDs_i)
            out_recs_list.append(rec_i)

        cA_batch = torch.stack(out_cA_list)
        cDs_batch = torch.stack(out_cDs_list)
        recs_batch = torch.stack(out_recs_list)

        return cA_batch, cDs_batch, recs_batch
