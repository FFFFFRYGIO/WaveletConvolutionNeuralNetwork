"""Pytorch Wavelet’s library usage example
source: https://pytorch-wavelets.readthedocs.io/en/latest/"""
import numpy as np
import torch

from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)


def main():
    """Run example instructions for pytorch_wavelets module."""

    xfm = DWTForward(J=3, mode='zero', wave='db3')  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode='zero', wave='db3')

    X = torch.randn(10, 5, 64, 64)

    Yl, Yh = xfm(X)
    print(Yl.shape)
    print(Yh[0].shape)
    print(Yh[1].shape)
    print(Yh[2].shape)

    Y = ifm((Yl, Yh))

    np.testing.assert_array_almost_equal(Y.cpu().numpy(), X.cpu().numpy())


if __name__ == '__main__':
    main()
