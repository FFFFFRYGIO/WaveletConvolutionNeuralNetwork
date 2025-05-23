import os
import urllib.request
import zipfile

import scipy.io
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Load data

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
test_labels_onehot = nn.functional.one_hot(test_labels_idx, num_classes=num_classes).float()

# Create tensors

train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels_onehot, dtype=torch.float32)

test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels_onehot, dtype=torch.float32)

# Wrap tensors in Dataset/DataLoader

train_data_tensor = train_data_tensor.unsqueeze(1)  # Add a dimension at dim=1
test_data_tensor = test_data_tensor.unsqueeze(1)  # Add a dimension at dim=1

empty_t1 = torch.empty(train_data_tensor.shape[0], 1, train_data_tensor.shape[2])
empty_t2 = torch.empty(train_data_tensor.shape[0], 1, train_data_tensor.shape[2])
empty_t3 = torch.empty(test_data_tensor.shape[0], 1, test_data_tensor.shape[2])
empty_t4 = torch.empty(test_data_tensor.shape[0], 1, test_data_tensor.shape[2])

train_ds = TensorDataset(torch.cat([train_data_tensor, empty_t1, empty_t2], dim=1), train_labels_tensor)
val_ds = TensorDataset(torch.cat([test_data_tensor, empty_t3, empty_t4], dim=1), test_labels_tensor)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

train_tensors = train_ds.tensors       # typically a tuple like (data_tensor, target_tensor)
val_tensors   = val_ds.tensors

torch.save(train_tensors, "train_tensors.pt")
torch.save(val_tensors,   "val_tensors.pt")
