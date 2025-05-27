"""This script is running both DWT and IDWT using both PyWavelets and pytorch_wavelets
to compare if their results matches."""

import pywt
import torch
from torch import nn
from pytorch_wavelets import DWT1D, IDWT1D


def get_dwt_pywt(signal: torch.Tensor, wavelet_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Run PyWavelets DWT for tensor."""
    wavelet = pywt.Wavelet(wavelet_name)
    sig_np = signal.squeeze().cpu().numpy()
    cA_np, cD_np = pywt.dwt(sig_np, wavelet)
    cA, cD = map(lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(signal), (cA_np, cD_np))
    return cA, cD


def get_idwt_pywt(cA: torch.Tensor, cD: torch.Tensor, wavelet_name: str) -> torch.Tensor:
    """Run PyWavelets IDWT for tensor."""
    wavelet = pywt.Wavelet(wavelet_name)
    cA_np = cA.squeeze().cpu().numpy()
    cD_np = cD.squeeze().cpu().numpy()
    rec_np = pywt.waverec([cA_np, cD_np], wavelet)
    rec = torch.from_numpy(rec_np).unsqueeze(0).unsqueeze(0).to(device=cA.device, dtype=cA.dtype)
    return rec


def torch_qmf(wavelet_filter: torch.Tensor) -> torch.Tensor:
    """Returns the Quadrature Mirror Filter (QMF) of the input tensor. Refactored from PyWavelets library:
    https://pywavelets.readthedocs.io/en/latest/ref/other-functions.html#pywt.qmf"""
    if wavelet_filter.dim() != 1:
        raise ValueError("`wavelet_filter` must be a 1-D tensor.")
    qm_filter = torch.flip(wavelet_filter, dims=[0])
    qm_filter[1::2] = -qm_filter[1::2]
    return qm_filter


def torch_orthogonal_filter_bank(
        scaling_filter: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns the orthogonal filter bank for the given scaling filter tensor. Refactored from PyWavelets library:
    https://pywavelets.readthedocs.io/en/latest/ref/other-functions.html#pywt.orthogonal_filter_bank"""
    if scaling_filter.dim() != 1:
        raise ValueError("`scaling_filter` must be a 1-D tensor.")
    length = scaling_filter.size(0)
    if length % 2 != 0:
        raise ValueError("`scaling_filter` length has to be even.")

    scaling_filter = scaling_filter.to(torch.float64)

    rec_lo = torch.sqrt(torch.tensor(2.0, dtype=scaling_filter.dtype)) * scaling_filter / scaling_filter.sum()
    dec_lo = torch.flip(rec_lo, dims=[0])

    rec_hi = torch_qmf(rec_lo)
    dec_hi = torch.flip(rec_hi, dims=[0])

    return dec_lo, dec_hi, rec_lo, rec_hi


def get_dwt_pytorch(signal: torch.Tensor, scaling_filter: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DWT using pytorch_wavelets."""
    dec_lo, dec_hi, rec_lo, rec_hi = torch_orthogonal_filter_bank(scaling_filter)
    dwt1d = DWT1D(wave=(dec_lo.tolist(), dec_hi.tolist()), mode='symmetric')
    cA, cD_list = dwt1d(signal)
    cD = cD_list[0]
    return cA, cD


def get_idwt_pytorch(cA: torch.Tensor, cD: torch.Tensor, scaling_filter: torch.Tensor) -> torch.Tensor:
    """Run IDWT using pytorch_wavelets."""
    dec_lo, dec_hi, rec_lo, rec_hi = torch_orthogonal_filter_bank(scaling_filter)
    idwt1d = IDWT1D(wave=(rec_lo.tolist(), rec_hi.tolist()), mode='symmetric')
    rec = idwt1d((cA, [cD]))
    return rec


if __name__ == '__main__':
    wavelet_main_name = 'db4'

    example_signal = torch.randn(1, 1, 128)

    # PyWavelets
    cA_pywt, cD_pywt = get_dwt_pywt(example_signal, wavelet_main_name)
    rec_pywt = get_idwt_pywt(cA_pywt, cD_pywt, wavelet_main_name)

    # Get rec_lo filter to parse it to pytorch_wavelets
    rec_lo_filter = pywt.Wavelet(wavelet_main_name).rec_lo
    rec_lo_tensor = nn.Parameter(torch.tensor(rec_lo_filter, dtype=torch.float32), requires_grad=True)

    # PyTorch-wavelets
    cA_t, cD_t = get_dwt_pytorch(example_signal, rec_lo_tensor)
    rec_t = get_idwt_pytorch(cA_t, cD_t, rec_lo_tensor)

    torch.testing.assert_close(cA_pywt, cA_t, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cD_pywt, cD_t, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(rec_pywt, rec_t, rtol=1e-6, atol=1e-6)

    print("✅ All DWT / IDWT match!")
