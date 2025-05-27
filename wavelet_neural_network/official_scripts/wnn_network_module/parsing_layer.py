"""Wavelet parsing layer class for Wavelet Neural Network."""
import torch
from torch import nn

from ..logger import logger


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
        batch_size = x3.size(0)
        x3_flat_with_padding = x3.view(batch_size, -1)

        non_filler_mask = torch.ne(x3_flat_with_padding, self.filler_value)
        x3_flat = x3_flat_with_padding.masked_select(non_filler_mask)

        counts = non_filler_mask.sum(dim=1)
        if not torch.all(counts == counts[0]):
            raise RuntimeError(f"Unequal non-filler counts: {counts.tolist()}")

        out = x3_flat.view(batch_size, int(counts[0].item()))

        return out
