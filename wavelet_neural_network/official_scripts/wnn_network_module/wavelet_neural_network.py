"""Wavelet Neural Network PyTorch class."""
from math import sqrt

import pywt
import torch
from torch import nn
from wnn_network_module.convolution_dwt_layer import WaveletDWTLayer
from wnn_network_module.parsing_layer import WaveletParsingLayer


class WaveletNeuralNet(nn.Module):
    """Wavelet Neural Network PyTorch class."""

    def calculate_dense_layers(self) -> zip:
        """Calculate input and output amount for all dense layers."""
        inputs = [self.signal_length * (self.convolution_layers_num - 1)]
        outputs = []

        for _ in range(self.dense_layers_num - 1):
            new_output = round(sqrt(inputs[-1]))
            outputs.append(new_output)
            inputs.append(new_output)
        outputs.append(self.num_classes)

        return zip(inputs, outputs)

    def __init__(
            self, num_classes: int, signal_length: int,
            wavelet_name: str = 'db4', num_dense_layers: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.signal_length = signal_length
        self.wavelet_name = wavelet_name
        filter_len = pywt.Wavelet(wavelet_name).dec_len
        self.convolution_layers_num = pywt.dwt_max_level(signal_length, filter_len)
        self.dense_layers_num = num_dense_layers

        self.conv_layers = nn.ModuleList([
            WaveletDWTLayer(wavelet_name, layer_number=i + 1)
            for i in range(self.convolution_layers_num)
        ])

        self.parsing_layer = WaveletParsingLayer()

        self.dense_layers = nn.ModuleList([
            nn.Linear(inp, out)
            for inp, out in self.calculate_dense_layers()
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
