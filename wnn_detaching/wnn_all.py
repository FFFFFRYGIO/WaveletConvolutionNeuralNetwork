# Import libraries
import urllib

import matplotlib
import numpy as np
import pywt

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary

from copy import deepcopy
from math import ceil, sqrt
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, confusion_matrix

# Load data

import os
import scipy.io

import urllib.request
import zipfile

DATA_FILE = 'ECGData'

zip_path = f'{DATA_FILE}.zip'
mat_path = f'{DATA_FILE}.mat'
download_url = 'https://github.com/mathworks/physionet_ECG_data/raw/main/ECGData.zip'

# 1. Download the ZIP if it doesn’t already exist
if not os.path.exists(zip_path):
    print(f"Downloading {zip_path}...")
    urllib.request.urlretrieve(download_url, zip_path)
    print("Download complete.")
else:
    print(f"{zip_path} already exists, skipping download.")

# 2. Extract the .mat (and any other) files if not already present
if not os.path.exists(mat_path):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        # by default, ZipFile.extractall won’t overwrite existing files
        z.extractall()
    print("Extraction complete.")
else:
    print(f"{mat_path} already exists, skipping extraction.")

mat_data = scipy.io.loadmat(f'{DATA_FILE}.mat')
raw_data = mat_data[DATA_FILE]
print(f'Number of signal from {DATA_FILE}: {len(raw_data["Data"][0, 0])}')

source_signals = {'ARR': [], 'CHF': [], 'NSR': []}

signals, labels = raw_data[0, 0]

for source_signal, label in zip(signals, labels):
  signal_type = label[0][0]
  source_signals[signal_type].append(source_signal)

print(f'Source signals labeled as ARR: {len(source_signals["ARR"])}')
print(f'Source signals labeled as CHF: {len(source_signals["CHF"])}')
print(f'Source signals labeled as NSR: {len(source_signals["NSR"])}')

# TEMPORARY - CUT THE DATA

arr_10_len = len(source_signals["ARR"]) // 3
source_signals["ARR"] = source_signals["ARR"][:arr_10_len]
chf_10_len = len(source_signals["CHF"]) // 3
source_signals["CHF"] = source_signals["CHF"][:chf_10_len]
nsr_10_len = len(source_signals["NSR"]) // 3
source_signals["NSR"] = source_signals["NSR"][:nsr_10_len]

print(f'Source signals labeled as ARR: {len(source_signals["ARR"])}')
print(f'Source signals labeled as CHF: {len(source_signals["CHF"])}')
print(f'Source signals labeled as NSR: {len(source_signals["NSR"])}')

# Create signals of desired length

from scipy.signal import medfilt

SOURCE_SIGNAL_LENGTH = len(source_signals['ARR'][0])
SOURCE_SIGNAL_FREQUENCY = SIGNAL_FREQUENCY = 128
SOURCE_SIGNAL_TIME = SOURCE_SIGNAL_LENGTH // SOURCE_SIGNAL_FREQUENCY

SIGNAL_TIME = 1  # Seconds
SIGNAL_LENGTH = SOURCE_SIGNAL_LENGTH // (SOURCE_SIGNAL_TIME // SIGNAL_TIME)
print(f'{SIGNAL_LENGTH=}')

desired_signals = {'ARR': [], 'CHF': [], 'NSR': []}

def denoise_signal(signal) -> None:
    kernel_size = int(1.0 * SIGNAL_FREQUENCY) | 1
    baseline = medfilt(signal, kernel_size=kernel_size)
    return signal - baseline

def normalize_signal(signal):
    peak = max(signal)
    return signal / peak

for signal_type, source_signals_list in source_signals.items():

  for source_signal in source_signals_list:
    signals_list = []

    for signal_number in range(source_signal.size // SIGNAL_LENGTH):
      start_index = signal_number * SIGNAL_LENGTH
      end_index = (signal_number + 1) * SIGNAL_LENGTH
      new_signal = source_signal[start_index:end_index]
      signal_denoised = denoise_signal(new_signal)
      signal_normalized = normalize_signal(signal_denoised)
      signals_list.append(signal_normalized)

    desired_signals[signal_type] += signals_list

print(f'Signals labeled as ARR: {len(desired_signals["ARR"])}')
print(f'Signals labeled as CHF: {len(desired_signals["CHF"])}')
print(f'Signals labeled as NSR: {len(desired_signals["NSR"])}')

# Create random test and train datasets

import random

random.seed(1234)

train_frac, test_frac = 0.8, 0.2
assert abs(train_frac + test_frac - 1.0) < 1e-9

train_data, train_labels = [], []
test_data, test_labels = [], []

for signal_type, signals_list in desired_signals.items():
    # shuffle within-class
    random.shuffle(signals_list)

    # split index
    split_idx = int(train_frac * len(signals_list))
    train_samples = signals_list[:split_idx]
    test_samples = signals_list[split_idx:]

    # accumulate
    train_data += train_samples
    train_labels += [signal_type] * len(train_samples)
    test_data += test_samples
    test_labels += [signal_type] * len(test_samples)

# final shuffle of the combined sets
train_pairs = list(zip(train_data, train_labels))
random.shuffle(train_pairs)
train_data, train_labels = map(list, zip(*train_pairs))

test_pairs = list(zip(test_data, test_labels))
random.shuffle(test_pairs)
test_data, test_labels = map(list, zip(*test_pairs))

assert len(train_data) == len(train_labels)
assert len(test_data) == len(test_labels)

print(f'Train data size: {len(train_data)}')
print(f'Test data size:  {len(test_data)}')

# Parse data to tensor

batch_size = 32

# Create one-hot encoding for labels tensors
unique_classes = sorted(set(train_labels + test_labels))
class_to_index = {cls: i for i, cls in enumerate(unique_classes)}

train_labels_idx = torch.tensor(
    [class_to_index[label] for label in train_labels],
    dtype=torch.long
)
test_labels_idx = torch.tensor(
    [class_to_index[label] for label in test_labels],
    dtype=torch.long
)

num_classes = len(unique_classes)
train_labels_onehot = nn.functional.one_hot(train_labels_idx, num_classes=num_classes).float()
test_labels_onehot = nn.functional.one_hot(test_labels_idx,  num_classes=num_classes).float()

# Create tensors

train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels_onehot, dtype=torch.float32)

test_data_tensor  = torch.tensor(test_data,  dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels_onehot, dtype=torch.float32)

# Wrap tensors in Dataset/DataLoader

train_data_tensor = train_data_tensor.unsqueeze(1)  # Add a dimension at dim=1
test_data_tensor = test_data_tensor.unsqueeze(1)    # Add a dimension at dim=1

empty_t1 = torch.empty(train_data_tensor.shape[0], 1, train_data_tensor.shape[2])
empty_t2 = torch.empty(train_data_tensor.shape[0], 1, train_data_tensor.shape[2])
empty_t3 = torch.empty(test_data_tensor.shape[0], 1, test_data_tensor.shape[2])
empty_t4 = torch.empty(test_data_tensor.shape[0], 1, test_data_tensor.shape[2])

train_ds = TensorDataset(torch.cat([train_data_tensor, empty_t1, empty_t2], dim=1), train_labels_tensor)
val_ds   = TensorDataset(torch.cat([test_data_tensor, empty_t3, empty_t4], dim=1),  test_labels_tensor)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=True)

# Create Wavelet Neural Network

class WaveletDWTLayer(nn.Module):
    """Create initial parameters bassed on given discrete wavelet name."""
    def __init__(self, wavelet_name: str = 'db4', layer_number: int = 0, filler_value: float = 10.1) -> None:
        super().__init__()
        self.wavelet_name = wavelet_name
        self.wavelet = pywt.Wavelet(self.wavelet_name)
        initial_scaling_filter = deepcopy(self.wavelet.dec_lo)
        self.params = nn.Parameter(torch.tensor(initial_scaling_filter, dtype=torch.float32))
        self.layer_number = layer_number
        self.filler_value = filler_value  # Out from <-10,10> range, to tell that this is only filler

    def update_wavelet(self) -> None:
        """Update wavelet with new filter based on parameters."""
        # copied_params = deepcopy(self.params)
        # scaling_filter_lo = copied_params.cpu().detach().numpy()
        # filters = pywt.orthogonal_filter_bank(scaling_filter_lo)
        # wavelet = pywt.Wavelet(f'cust_{self.wavelet_name}', filter_bank=filters)
        # wavelet.orthogonal = True
        # wavelet.biorthogonal = True

        # diffs = [x - y for x, y in zip(wavelet.dec_lo, self.wavelet.dec_lo)]
        # if any(d != 0 for d in diffs):
        #     print('diffs:', diffs)

        # self.wavelet = wavelet
        pass

    def run_dwt(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run DWT on input signal."""
        cA, cD = pywt.dwt(signal, self.wavelet)
        return cA, cD

    def run_idwt(self, cA: np.ndarray, cD: np.ndarray, cDs: list[np.ndarray]) -> np.ndarray:
        """Run inverse DWT based on coefficients list."""

        print('cDs', type(cDs), len(cDs), type(cDs[0]), cDs[0].shape)

        included_cDs = [np.zeros_like(cA), cD]
        print('run_idwt included_cDs', included_cDs)

        print('run_idwt len included_cDs', [len(x) for x in included_cDs])

        for coeff_number, coeff_with_pad in enumerate(cDs[::-1]):
            coeff = coeff_with_pad[coeff_with_pad != self.filler_value]
            print('run_idwt len coeff', coeff_with_pad.shape, len(coeff))
            if coeff_number == 0:
                included_cDs.append(coeff)
            else:
                included_cDs.append(np.zeros_like(coeff))

        print('run_idwt len included_cDs', [len(x) for x in included_cDs])
        R = pywt.waverec(included_cDs, self.wavelet)
        print('run_idwt len included_cDs', len(R), [len(x) for x in included_cDs])
        return R

    def run_wavelet_computation(self,
        signal: np.ndarray,
        cDs: list[np.ndarray],
        reconstructions: list[np.ndarray]
        ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Run DWT and IDWT and return result for batch element."""

        print('cDs', type(cDs), len(cDs), type(cDs[0]), cDs[0].shape)

        # Create new wavelet with current parameters
        self.update_wavelet()

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

        # Process each element in the batch individually
        cA_batch, cD_batch, R_batch = [], [], []

        for i in range(signal_tensor.shape[0]):
            signal = signal_tensor[i, :].cpu().detach().numpy()
            cDs = coeffs_tensor[i, :].cpu().detach().numpy()
            reconstructions = reconstructions_tensor[i, :].cpu().detach().numpy()

            signal = signal.reshape(-1)
            cDs = [cD for cD in cDs]
            reconstructions = [rec for rec in reconstructions]

            print('cDs', type(cDs), len(cDs), type(cDs[0]), cDs[0].shape)
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
    def __init__(self, filler_value: float = 10.1) -> None:
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
                fully_reconstructed.append(filtered)

            print('WaveletParsingLayer', [len(x) for x in fully_reconstructed])

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
            WaveletDWTLayer(wavelet_name, layer_number=i+1)
            for i in range(max_level)
        ])

        self.parsing_layer = WaveletParsingLayer()

        inputs = [self.get_parsing_layer_len(signal_length, max_level)]
        outputs = []
        for _ in range(num_dense_layes - 1):
            new_output = round(sqrt(inputs[-1]))
            outputs.append(new_output)
            inputs.append(new_output)
        outputs.append(num_classes)

        print('WaveletNauralNet', inputs, outputs)

        self.dense_layers = nn.ModuleList([
            nn.Linear(inp, out)
            for inp, out in zip(inputs, outputs)
        ])

    def get_parsing_layer_len(self, signal_length: int, level: int = 0) -> int:
        """Calculate the length of the coefficient based on layer"""
        return signal_length * level
        # parsing_layer_len = 0
        # next_signal_length = signal_length
        # for _ in range(level):
        #     print('get_parsing_layer_len', next_signal_length)
        #     parsing_layer_len += next_signal_length
        #     next_signal_length = (next_signal_length + self.filter_len - 1) // 2
        # parsing_layer_len += next_signal_length
        # print('get_parsing_layer_len', next_signal_length)
        # print('get_parsing_layer_len', next_signal_length)
        # return parsing_layer_len

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

def train_and_validate(model, train_loader, val_loader, epochs, lr, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses   = []
    val_losses     = []
    val_accs       = []
    val_precisions = []
    val_confmats   = []
    first_layers_params_monitor = []
    last_layer_params_monitor = []

    # model.to(device)

    for epoch in range(epochs):
        current_first_layer_params = model.conv_layers[0].params.cpu().detach().numpy()
        first_layers_params_monitor.append(current_first_layer_params)
        try:
            current_last_layer_params = model.dense_layers[-1].weight.cpu().detach().numpy()
        except AttributeError as e:
            print('Skipping,', e)
            current_last_layer_params = [None] * len(current_first_layer_params)
        last_layer_params_monitor.append(current_first_layer_params)

        # — TRAIN —
        model.train()
        running_train_loss = 0.0
        for data, target in train_loader:
            # data, target = data.to(device), target.to(device)
            # split channels if needed
            x1, x2, x3 = data[:,0:1,:], data[:,1:2,:], data[:,2:3,:]

            optimizer.zero_grad()
            logits = model(x1, x2, x3)
            train_loss = criterion(logits, target.argmax(dim=1))
            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # — VALIDATE —
        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for data, target in val_loader:
                # data, target = data.to(device), target.to(device)
                x1, x2, x3 = data[:,0:1,:], data[:,1:2,:], data[:,2:3,:]

                logits = model(x1, x2, x3)
                val_loss = criterion(logits, target.argmax(dim=1))

                preds = logits.argmax(dim=1)

                running_val_loss += val_loss.item()

                all_preds.append(preds.cpu())
                all_trues.append(target.argmax(dim=1).cpu())

        # concatenate across batches
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_trues).numpy()

        # accuracy, precision, confusion matrix
        epoch_acc   = (y_pred == y_true).mean()
        epoch_prec  = precision_score(y_true, y_pred, average='macro', zero_division=0)
        conf_mat    = confusion_matrix(y_true, y_pred)

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        val_accs.append(epoch_acc)
        val_precisions.append(epoch_prec)
        val_confmats.append(conf_mat)

        print(f"Epoch {epoch+1:03d}/{epochs:03d} — "
              f"Train loss: {epoch_train_loss:.4f}, "
              f"Val loss: {epoch_val_loss:.4f}, "
              f"Val Acc: {epoch_acc:.4f}, "
              f"Val Prec: {epoch_prec:.4f}, "
              f"Confusion Matrix: {conf_mat.tolist()}, "
              f"1st layer params: {current_first_layer_params[0:4]}, "
              f"last layer params: {current_last_layer_params[0][0:4]}, ")

    return {
        'train_losses':   train_losses,
        'val_losses':       val_losses,
        'val_accs':       val_accs,
        'val_precisions': val_precisions,
        'val_confmats':   val_confmats,
        'first_layers_params_monitor': first_layers_params_monitor,
        'last_layer_params_monitor': last_layer_params_monitor,
    }

model = WaveletNauralNet(num_classes=num_classes)

# Example usage:
results = train_and_validate(
    model, train_loader, val_loader,
    epochs=100, lr=1e-3
)

# Plotting Loss, Accuracy, Precision
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

fig, axes = plt.subplots(
    nrows=4, ncols=2, figsize=(10,4))

axes[0, 0].plot(results['train_losses'])
axes[0, 0].set_title("Training Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")

axes[0, 1].plot(results['val_losses'])
axes[0, 1].set_title("Validation Loss")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")

axes[1, 0].plot(results['val_accs'])
axes[1, 0].set_title("Validation Accuracy")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Accuracy")

axes[1, 1].plot(results['val_precisions'])
axes[1, 1].set_title("Validation Precision (macro)")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Precision")

first_layer_params_means = [np.mean(params) for params in results['first_layers_params_monitor']]
axes[2, 0].plot(first_layer_params_means)
axes[2, 0].set_title("Mean values of first layer parameters")
axes[2, 0].set_xlabel("Epoch")
axes[2, 0].set_ylabel("Mean value")

first_layer_params_stds  = [np.std(params)  for params in results['first_layers_params_monitor']]
axes[2, 1].plot(first_layer_params_stds)
axes[2, 1].set_title("Standard deviation of first layer parameters")
axes[2, 1].set_xlabel("Epoch")
axes[2, 1].set_ylabel("Std value")

last_layer_params_means = [np.mean(params) for params in results['last_layer_params_monitor']]
axes[3, 0].plot(last_layer_params_means)
axes[3, 0].set_title("Mean values of last layer parameters")
axes[3, 0].set_xlabel("Epoch")
axes[3, 0].set_ylabel("Mean value")

last_layer_params_stds  = [np.std(params)  for params in results['last_layer_params_monitor']]
axes[3, 1].plot(last_layer_params_stds)
axes[3, 1].set_title("Standard deviation of last layer parameters")
axes[3, 1].set_xlabel("Epoch")
axes[3, 1].set_ylabel("Std value")

plt.tight_layout()
plt.show()

# To inspect the final confusion matrix:
print("Final Confusion Matrix:")
print(results['val_confmats'][-1])