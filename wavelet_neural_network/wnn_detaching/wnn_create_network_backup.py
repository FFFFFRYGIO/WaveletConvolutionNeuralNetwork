from copy import deepcopy
from math import sqrt

import numpy as np
import pywt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary

# Load data

SIGNAL_LENGTH = 128
num_classes = 3
batch_size = 32

loaded_train_data, loaded_train_labels = torch.load("train_tensors.pt")
loaded_val_data, loaded_val_labels = torch.load("val_tensors.pt")

# Re-wrap them in TensorDatasets
train_ds = TensorDataset(loaded_train_data, loaded_train_labels)
val_ds = TensorDataset(loaded_val_data, loaded_val_labels)

# Finally, recreate your DataLoaders
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

# Create Wavelet Neural Network


class WaveletDWTLayer(nn.Module):
    """Create initial parameters based on given discrete wavelet name."""

    def __init__(self, wavelet_name: str = 'db4', layer_number: int = 0, filler_value: float = 1.1) -> None:
        super().__init__()
        self.wavelet_name = wavelet_name
        self.wavelet = pywt.Wavelet(self.wavelet_name)
        initial_scaling_filter = deepcopy(self.wavelet.rec_lo)
        self.weights = nn.Parameter(torch.tensor(initial_scaling_filter, dtype=torch.float32), requires_grad=True)
        self.layer_number = layer_number
        self.filler_value = filler_value  # Out from <-1,1> range, to tell that this is only a filler

    def update_wavelet(self) -> None:
        """Update wavelet with new filter based on parameters."""
        pass
        # copied_params = deepcopy(self.params.tolist())
        # scaling_filter_lo = copied_params
        # filters = pywt.orthogonal_filter_bank(scaling_filter_lo)
        # wavelet = pywt.Wavelet(f'cust_{self.wavelet_name}', filter_bank=filters)
        # wavelet.orthogonal = True
        # wavelet.biorthogonal = True
        #
        # diffs = [x - y for x, y in zip(wavelet.rec_lo, self.wavelet.rec_lo)]
        # if any(d != 0 for d in diffs):
        #     print('update_wavelet diffs:', diffs)
        #
        # self.wavelet = wavelet

    def run_dwt(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run DWT on input signal."""
        cA, cD = pywt.dwt(signal, self.wavelet)
        return cA, cD

    def run_idwt(self, cA: np.ndarray, cD: np.ndarray, cDs: list[np.ndarray]) -> np.ndarray:
        """Run inverse DWT based on a coefficient list."""
        included_cDs = [np.zeros_like(cA), cD]

        for coeff_number, coeff_with_pad in enumerate(cDs[::-1]):
            coeff = coeff_with_pad[coeff_with_pad != self.filler_value]
            if coeff_number == 0:
                included_cDs.append(coeff)
            else:
                included_cDs.append(np.zeros_like(coeff))

        R = pywt.waverec(included_cDs, self.wavelet)
        return R

    def run_wavelet_computation(self,
                                signal: np.ndarray,
                                cDs: list[np.ndarray],
                                reconstructions: list[np.ndarray]
                                ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Run DWT and IDWT and return result for a batch element."""

        # Run DWT
        cA, cD = self.run_dwt(signal)

        if not self.layer_number == 1:
            # Run IDWT
            R = self.run_idwt(cA, cD, cDs)

        # Add cD to cDs
        if self.layer_number == 1:
            cDs = np.array([cD])
        else:
            cD_pad_amount = cDs[0].size - cD.size
            cD_padded = np.pad(cD, (0, cD_pad_amount), constant_values=self.filler_value)
            cDs = np.vstack([cDs, cD_padded])

        # Add new reconstruction
        if not self.layer_number == 1:
            if self.layer_number == 2:
                reconstructions = np.array([R])
            else:
                reconstructions = np.vstack([reconstructions, R])

        return cA, cDs, reconstructions

    def forward(
            self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signal_tensor = x1
        coeffs_tensor = x2
        reconstructions_tensor = x3

        # Create new wavelet with current parameters
        self.update_wavelet()

        # Process each element in the batch individually
        cA_batch, cD_batch, R_batch = [], [], []

        for i in range(signal_tensor.shape[0]):
            signal = signal_tensor[i, :].cpu().detach().numpy()
            cDs = coeffs_tensor[i, :].cpu().detach().numpy()
            reconstructions = reconstructions_tensor[i, :].cpu().detach().numpy()

            signal = signal.reshape(-1)
            cDs = [cD for cD in cDs]
            reconstructions = [rec for rec in reconstructions]

            new_cA, new_cDs, new_rec = self.run_wavelet_computation(signal, cDs, reconstructions)

            # Add element to batch segment
            cA_batch.append(new_cA)
            cD_batch.append(new_cDs)
            R_batch.append(new_rec)

        # Stack the results back into batches
        cA_np = np.stack(cA_batch)
        cD_np = np.stack(cD_batch)
        R_np = np.stack(R_batch)

        # Convert NumPy arrays back to PyTorch tensors and require gradients
        x1_r = torch.tensor(cA_np, dtype=x1.dtype, device=x1.device, requires_grad=True)
        x2_r = torch.tensor(cD_np, dtype=x2.dtype, device=x2.device, requires_grad=True)
        x3_r = torch.tensor(R_np, dtype=x3.dtype, device=x3.device, requires_grad=True)

        return (x1_r, x2_r, x3_r)


class WaveletParsingLayer(nn.Module):
    """Layer that parses convolution result to dense layer."""

    def __init__(self, filler_value: float = 1.1) -> None:
        super().__init__()
        self.filler_value = filler_value

    def forward(
            self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> torch.Tensor:
        # Convert the flattened reconsructions to a PyTorch tensor
        reconstructions_tensor = x3
        R_batch = []
        for i in range(reconstructions_tensor.shape[0]):
            fully_reconstructed = []
            reconstructions = reconstructions_tensor[i, :].cpu().detach().numpy()

            for reconstruction in reconstructions:
                filtered = reconstruction[reconstruction != self.filler_value]
                fully_reconstructed.extend(filtered)

            R_batch.append(fully_reconstructed)

        R_np = np.stack(R_batch)

        x3_r = torch.tensor(R_np, dtype=x3.dtype, device=x3.device, requires_grad=True)

        return x3_r


class WaveletNauralNet(nn.Module):
    def __init__(
            self, num_classes: int, signal_length: int = SIGNAL_LENGTH,
            wavelet_name: str = 'db4', num_dense_layes: int = 3) -> None:
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
        for _ in range(num_dense_layes - 1):
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
        o1, o2, o3 = x1, x2, x3
        for layer in self.conv_layers:
            o1, o2, o3 = layer(o1, o2, o3)

        dense_res = self.parsing_layer(o1, o2, o3)

        for layer in self.dense_layers:
            normalised_dense_res = nn.functional.relu(dense_res)
            dense_res = layer(normalised_dense_res)

        return dense_res


# View test model

test_model = WaveletNauralNet(num_classes=num_classes)
print(test_model)

total_params = sum(p.numel() for p in test_model.parameters())
trainable_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
print(f"Total params and trainable params: {total_params} / {trainable_params}")

one_batch_data, _ = next(iter(train_loader))

x1_summary = one_batch_data[:, 0:1, :]
x2_summary = one_batch_data[:, 1:2, :]
x3_summary = one_batch_data[:, 2:3, :]

summary(test_model, input_data=(x1_summary, x2_summary, x3_summary),
        col_names=["input_size", "output_size", "num_params", "trainable"])
