"""Wavelet parsing layer class for Wavelet Neural Network."""
import numpy as np
import torch
from torch import nn

from logger import logger


class WaveletParsingLayer(nn.Module):
    """Layer that parses convolution results to dense layer."""

    def __init__(self, filler_value: float = 10.1) -> None:
        super().__init__()
        self.filler_value = filler_value  # Out from <-10,10> range, to tell that this is only a filler

    def forward(
            self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> torch.Tensor:
        """Convert the flattened reconstructions to a PyTorch tensor"""
        logger.debug('{module_name} forward'.format(module_name=self.__class__.__name__))
        reconstructions_tensor = x3

        R_batch = []
        for i in range(reconstructions_tensor.shape[0]):
            fully_reconstructed = []
            reconstructions_listed = reconstructions_tensor[i, :].tolist()
            reconstructions = [np.asarray(rec) for rec in reconstructions_listed]

            for reconstruction in reconstructions:
                filtered = reconstruction[reconstruction != self.filler_value]
                fully_reconstructed.extend(filtered)

            R_batch.append(fully_reconstructed)

        R_np = np.stack(R_batch)

        x3_r = torch.tensor(R_np, dtype=x3.dtype, device=x3.device, requires_grad=True)

        return x3_r
