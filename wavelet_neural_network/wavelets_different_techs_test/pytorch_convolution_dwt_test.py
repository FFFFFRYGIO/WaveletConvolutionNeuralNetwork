"""This script is running both DWT using both PyWavelets and torch.nn.Conv1d to compare if their results matches."""
import pywt
import torch
import torch.nn.functional as F
from torch import nn


def get_dwt_pywt(signal: torch.Tensor, wavelet_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Run PyWavelets DWT for tensor."""
    wavelet = pywt.Wavelet(wavelet_name)
    sig_np = signal.squeeze().cpu().numpy()
    cA_np, cD_np = pywt.dwt(sig_np, wavelet)
    cA, cD = map(lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(signal), (cA_np, cD_np))
    return cA, cD


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


def get_dwt_pytorch_conv1d(signal: torch.Tensor, scaling_filter: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DWT using torch.nn.Conv1d."""
    dec_lo, dec_hi, rec_lo, rec_hi = torch_orthogonal_filter_bank(scaling_filter)

    # 2) back to float32 (or whatever your signal is)
    dec_lo = dec_lo.to(signal.dtype)
    dec_hi = dec_hi.to(signal.dtype)

    # reshape to Conv1d kernels
    L = dec_lo.numel()
    w_lo = dec_lo.view(1, 1, L)
    w_hi = dec_hi.view(1, 1, L)

    # 3) symmetric pad both ends by P = L-1
    P = L - 1
    left = signal[..., :P].flip(-1)  # x[0..P-1] reversed
    right = signal[..., -P:].flip(-1)  # x[-P..-1] reversed
    x = torch.cat([left, signal, right], dim=-1)

    # 4) full “valid” conv (no decimation yet)
    convA = F.conv1d(x, w_lo, stride=1)
    convD = F.conv1d(x, w_hi, stride=1)

    # 5) down‐sample by picking the odd‐indexed outputs
    cA = convA[:, :, 1::2]
    cD = convD[:, :, 1::2]

    return cA, cD


if __name__ == '__main__':
    wavelet_main_name = 'db4'

    example_signal = torch.randn(1, 1, 128)

    # PyWavelets
    cA_pywt, cD_pywt = get_dwt_pywt(example_signal, wavelet_main_name)

    # Get rec_lo filter to parse it to pytorch_wavelets
    rec_lo_filter = pywt.Wavelet(wavelet_main_name).rec_lo
    rec_lo_tensor = nn.Parameter(torch.tensor(rec_lo_filter, dtype=torch.float32), requires_grad=True)

    # PyTorch torch.nn.Conv1d
    cA_t, cD_t = get_dwt_pytorch_conv1d(example_signal, rec_lo_tensor)

    torch.testing.assert_close(cA_pywt, cA_t, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cD_pywt, cD_t, rtol=1e-6, atol=1e-6)

    print("✅ All DWT / IDWT match!")
