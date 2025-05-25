"""Wavelet Neural Network PyTorch class."""
from math import sqrt

import pywt
import torch
from torch import nn
from wnn_network_module.convolution_dwt_layer import WaveletDWTLayer
from wnn_network_module.parsing_layer import WaveletParsingLayer


class WaveletNeuralNet(nn.Module):
    def __init__(
            self, num_classes: int, signal_length: int,
            wavelet_name: str = 'db4', num_dense_layers: int | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.filter_len = pywt.Wavelet(wavelet_name).dec_len
        max_level = pywt.dwt_max_level(signal_length, self.filter_len)
        self.conv_layers = nn.ModuleList([
            WaveletDWTLayer(wavelet_name, layer_number=i + 1)
            for i in range(max_level)
        ])

        self.parsing_layer = WaveletParsingLayer()

        inputs = [signal_length * (max_level - 1)]
        outputs = []

        if num_dense_layers is None:
            num_dense_layers = 3
        for _ in range(num_dense_layers - 1):
            new_output = round(sqrt(inputs[-1]))
            outputs.append(new_output)
            inputs.append(new_output)
        outputs.append(num_classes)

        self.dense_layers = nn.ModuleList([
            nn.Linear(inp, out)
            for inp, out in zip(inputs, outputs)
        ])

    def forward(
            self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize WNN layers."""
        o1, o2, o3 = x1, x2, x3
        for layer in self.conv_layers:
            o1, o2, o3 = layer(o1, o2, o3)

        dense_res = self.parsing_layer(o1, o2, o3)

        for layer in self.dense_layers:
            normalised_dense_res = nn.functional.relu(dense_res)
            dense_res = layer(normalised_dense_res)

        return dense_res
