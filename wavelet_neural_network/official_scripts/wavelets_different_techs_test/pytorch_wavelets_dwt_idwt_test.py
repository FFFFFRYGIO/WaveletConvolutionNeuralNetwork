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
    rec = torch.from_numpy(rec_np).unsqueeze(0).unsqueeze(0).to(signal)
    return rec


def get_dwt_pytorch(signal: torch.Tensor, wavelet_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DWT using pytorch_wavelets."""
    wavelet = pywt.Wavelet(wavelet_name)
    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dwt1d = DWT1D(wave=(dec_lo, dec_hi), mode='symmetric')
    cA, cD_list = dwt1d(signal)
    cD = cD_list[0]
    return cA, cD


def get_idwt_pytorch(cA: torch.Tensor, cD: torch.Tensor, wavelet_name: str) -> torch.Tensor:
    wavelet = pywt.Wavelet(wavelet_name)
    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    idwt1d = IDWT1D(wave=(rec_lo, rec_hi), mode='symmetric')
    rec = idwt1d((cA, [cD]))
    return rec


if __name__ == '__main__':
    wavelet_main_name = 'db4'

    example_signal = torch.randn(1, 1, 128)
    signal = example_signal

    # PyWavelets
    cA_pywt, cD_pywt = get_dwt_pywt(example_signal, wavelet_main_name)
    rec_pywt = get_idwt_pywt(cA_pywt, cD_pywt, wavelet_main_name)

    # PyTorch-wavelets
    cA_t, cD_t = get_dwt_pytorch(example_signal, wavelet_main_name)
    rec_t = get_idwt_pytorch(cA_t, cD_t, wavelet_main_name)

    torch.testing.assert_close(cA_pywt, cA_t, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cD_pywt, cD_t, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(rec_pywt, rec_t, rtol=1e-6, atol=1e-6)

    print("✅ All DWT / IDWT match!")
