import pywt
import torch
# from pytorch_wavelets import DWT1D, IDWT1D
from my_DWT1D.my_DWT1D import DWT1DForward as DWT1D, DWT1DInverse as IDWT1D
from torch import nn


def get_dwt_pywt(signal: torch.Tensor, wavelet_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    wavelet = pywt.Wavelet(wavelet_name)
    sig_np = signal.squeeze().cpu().numpy()  # → shape (L,)
    cA_np, cD_np = pywt.dwt(sig_np, wavelet)
    cA, cD = map(lambda x: torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(signal), (cA_np, cD_np))
    return cA, cD


def get_idwt_pywt(cA: torch.Tensor, cD: torch.Tensor, wavelet_name: str) -> torch.Tensor:
    wavelet = pywt.Wavelet(wavelet_name)
    cA_np = cA.squeeze().cpu().numpy()
    cD_np = cD.squeeze().cpu().numpy()
    rec_np = pywt.waverec([cA_np, cD_np], wavelet)
    rec = torch.from_numpy(rec_np).unsqueeze(0).unsqueeze(0).to(signal)
    return rec


def torch_qmf(filt: torch.Tensor) -> torch.Tensor:
    """
    Returns the Quadrature Mirror Filter (QMF) of the input tensor.

    Parameters
    ----------
    filt : torch.Tensor
        1-D input filter tensor.

    Returns
    -------
    torch.Tensor
        Quadrature mirror of the input filter.
    """
    if filt.dim() != 1:
        raise ValueError("`filt` must be a 1-D tensor.")
    # Reverse tensor
    qm_filter = torch.flip(filt, dims=[0])
    # Negate every other element starting from index 1
    qm_filter[1::2] = -qm_filter[1::2]
    return qm_filter


def torch_orthogonal_filter_bank(scaling_filter: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns the orthogonal filter bank for the given scaling filter tensor.

    The orthogonal filter bank consists of the HPFs and LPFs at
    decomposition and reconstruction stage for the input scaling filter.

    Parameters
    ----------
    scaling_filter : torch.Tensor
        1-D input scaling filter tensor (father wavelet).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        The orthogonal filter bank in the order:
        1) Decomposition LPF
        2) Decomposition HPF
        3) Reconstruction LPF
        4) Reconstruction HPF
    """
    if scaling_filter.dim() != 1:
        raise ValueError("`scaling_filter` must be a 1-D tensor.")
    length = scaling_filter.size(0)
    if length % 2 != 0:
        raise ValueError("`scaling_filter` length has to be even.")

    # Ensure double precision for numerical consistency
    scaling_filter = scaling_filter.to(torch.float64)

    # Reconstruction low-pass filter
    rec_lo = torch.sqrt(torch.tensor(2.0, dtype=scaling_filter.dtype)) * scaling_filter / scaling_filter.sum()
    # Decomposition low-pass: reverse of rec_lo
    dec_lo = torch.flip(rec_lo, dims=[0])

    # Reconstruction high-pass via QMF
    rec_hi = torch_qmf(rec_lo)
    # Decomposition high-pass: reverse of rec_hi
    dec_hi = torch.flip(rec_hi, dims=[0])

    return dec_lo, dec_hi, rec_lo, rec_hi



def get_dwt_pytorch(signal: torch.Tensor, rec_lo: nn.Parameter) -> tuple[torch.Tensor, torch.Tensor]:
    dec_lo, dec_hi, rec_lo, rec_hi = torch_orthogonal_filter_bank(rec_lo)

    dwt1d = DWT1D(wave=(dec_lo, dec_hi), mode='symmetric')
    cA, cD_list = dwt1d(signal)
    cD = cD_list[0]
    return cA, cD


def get_idwt_pytorch(cA: torch.Tensor, cD: torch.Tensor, rec_lo: nn.Parameter) -> torch.Tensor:
    dec_lo, dec_hi, rec_lo, rec_hi = torch_orthogonal_filter_bank(rec_lo)
    idwt1d = IDWT1D(wave=(rec_lo, rec_hi), mode='symmetric')
    rec = idwt1d((cA, [cD]))
    return rec


if __name__ == '__main__':
    wavelet_main_name = 'db4'

    signal = torch.randn(1, 1, 128)

    # PyWavelets
    cA_pywt, cD_pywt = get_dwt_pywt(signal, wavelet_main_name)
    rec_pywt = get_idwt_pywt(cA_pywt, cD_pywt, wavelet_main_name)

    # PyTorch-wavelets
    rec_lo_filter = pywt.Wavelet(wavelet_main_name).rec_lo
    rec_lo_tensor = nn.Parameter(torch.tensor(rec_lo_filter, dtype=torch.float32), requires_grad=True)

    cA_t, cD_t = get_dwt_pytorch(signal, rec_lo_tensor)
    rec_t = get_idwt_pytorch(cA_t, cD_t, rec_lo_tensor)

    torch.testing.assert_close(cA_pywt, cA_t, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cD_pywt, cD_t, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(rec_pywt, rec_t, rtol=1e-6, atol=1e-6)

    print("✅ All DWT / IDWT match exactly!")
