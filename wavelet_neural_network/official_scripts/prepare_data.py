"""Script with function to prepare data tensors for Wavelet Neural Network."""
import os
import random
import urllib.request
import zipfile

import numpy as np
import scipy
import torch
from scipy.signal import medfilt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from logger import logger


def download_dataset() -> tuple[str, str]:
    """Download and extract ECG signals file from the url."""

    download_url = os.getenv('DATA_SOURCE_URL')
    mat_file = download_url.split('/')[-1].split('.')[0]
    zip_path, mat_path = rf'data/{mat_file}.zip', rf'data/{mat_file}.mat'

    if not os.path.exists(zip_path):
        logger.info('Downloading {zip_path}.'.format(zip_path=zip_path))
        urllib.request.urlretrieve(download_url, zip_path)
        logger.info('Download complete.')
    else:
        logger.info('{zip_path} already exists, skipping download.'.format(zip_path=zip_path))

    if not os.path.exists(mat_path):
        logger.info('Extracting {zip_path}.'.format(zip_path=zip_path))
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(path='data')
        logger.info('Extraction complete.')
    else:
        logger.info('{mat_path} already exists, skipping download.'.format(mat_path=mat_path))

    return mat_path, mat_file


def split_signals_by_classes(signals_data: np.void) -> dict[str, list[np.array]]:
    """Split raw signals data set by its classes."""

    source_signals = {}

    signals, labels = signals_data

    for source_signal, label in zip(signals, labels):
        signal_type = label[0][0]
        if signal_type not in source_signals.keys():
            source_signals[signal_type] = []
        source_signals[signal_type].append(source_signal)

    for label, signals_list in source_signals.items():
        logger.info('Source signals labeled as {label}: {signals_list_length}'.format(
            label=label, signals_list_length=len(signals_list))
        )

    return source_signals


def split_signals_by_desired_length(
        raw_signals_by_classes: dict[str, list[np.array]], signal_time: int | None = None
) -> tuple[dict[str, list[np.array]], int]:
    """Split ECG signals on subsets of a desired time interval."""

    signal_frequency = int(os.getenv('ECG_DATA_SOURCE_SIGNAL_FREQUENCY'))

    first_signal = next(iter(raw_signals_by_classes.values()))[0]
    source_signal_length = len(first_signal)

    if signal_time is None:
        desired_signal_length = source_signal_length
    else:
        desired_signal_length = signal_frequency // signal_time

    desired_signals = {}

    for label, source_signals_list in raw_signals_by_classes.items():
        if label not in desired_signals.keys():
            desired_signals[label] = []

        signals_list = []
        for source_signal in source_signals_list:
            for signal_number in range(source_signal.size // desired_signal_length):
                start_index = signal_number * desired_signal_length
                end_index = (signal_number + 1) * desired_signal_length
                new_signal = source_signal[start_index:end_index]
                signals_list.append(new_signal)

        desired_signals[label] += signals_list

    for label, signals_list in desired_signals.items():
        logger.info('Desired signals labeled as {label}: {signals_list_length}'.format(
            label=label, signals_list_length=len(signals_list))
        )

    return desired_signals, desired_signal_length


def denoise_signal(signal: np.array, signal_frequency: int) -> np.array:
    """Remove noise from ECG signal."""
    raw_kernel_val = int(1.0 * signal_frequency)
    raw_kernel = raw_kernel_val if raw_kernel_val % 2 else raw_kernel_val + 1
    signal_len = len(signal)
    max_odd = signal_len if signal_len % 2 else signal_len - 1

    kernel_size = min(raw_kernel, max_odd)
    baseline = medfilt(signal, kernel_size=kernel_size)
    return signal - baseline


def normalize_signal_max_abs(signal: np.array) -> np.array:
    """Normalize signal using maximum value."""
    peak = max(signal)
    return signal / peak


def normalize_signals(signals_data: dict[str, list[np.array]]) -> dict[str, list[np.array]]:
    """Normalize signal by MinMax normalization and removing noise from signal."""

    signals_normalized = {}

    signal_frequency = int(os.getenv('ECG_DATA_SOURCE_SIGNAL_FREQUENCY'))

    for label, signals_list in signals_data.items():
        if label not in signals_normalized.keys():
            signals_normalized[label] = []

        for signal in signals_list:
            signal_denoised = denoise_signal(signal, signal_frequency)
            signal_normalized = normalize_signal_max_abs(signal_denoised)

            signals_normalized[label].append(signal_normalized)

    for label, signals_list in signals_normalized.items():
        logger.info('Normalized signals labeled as {label}: {signals_list_length}'.format(
            label=label, signals_list_length=len(signals_list))
        )

    return signals_normalized


def create_train_and_validation_subsets(
        signals_data: dict[str, list[np.array]]
) -> tuple[list[np.array], list[str], list[np.array], list[str]]:
    """Split ECG signal data to train and validation subsets."""
    train_frac = float(os.getenv('TRAIN_FRAC'))

    train_signals: list[np.array] = []
    train_labels: list[str] = []
    validation_signals: list[np.array] = []
    validation_labels: list[str] = []

    for signal_type, signals_list in signals_data.items():
        random.shuffle(signals_list)

        split_idx = int(train_frac * len(signals_list))
        train_samples = signals_list[:split_idx]
        validation_samples = signals_list[split_idx:]

        train_signals.extend(train_samples)
        train_labels.extend([signal_type] * len(train_samples))
        validation_signals.extend(validation_samples)
        validation_labels.extend([signal_type] * len(validation_samples))

    train_pairs = list(zip(train_signals, train_labels))
    random.shuffle(train_pairs)
    train_shuffled_signals, train_shuffled_labels = zip(*train_pairs)
    train_shuffled_signals = list(train_shuffled_signals)
    train_shuffled_labels = list(train_shuffled_labels)

    validation_pairs = list(zip(validation_signals, validation_labels))
    random.shuffle(validation_pairs)
    validation_shuffled_signals, validation_shuffled_labels = zip(*validation_pairs)
    validation_shuffled_signals = list(validation_shuffled_signals)
    validation_shuffled_labels = list(validation_shuffled_labels)

    assert len(train_shuffled_signals) == len(train_labels)
    assert len(validation_shuffled_signals) == len(validation_labels)

    logger.info('Train data size: {train_data_size}'.format(train_data_size=len(train_shuffled_signals)))
    logger.info('Validation data size: {val_data_size}'.format(val_data_size=len(validation_shuffled_signals)))

    return train_shuffled_signals, train_shuffled_labels, validation_shuffled_signals, validation_shuffled_labels


def create_train_and_validation_tensors_datasets(
        train_signals: list[np.array], train_labels: list[str], val_signals: list[np.array], val_labels: list[str]
) -> tuple[TensorDataset, TensorDataset, int]:
    """Create train and validation tensors from PyTorch library."""

    unique_classes = sorted(set(train_labels + val_labels))
    class_to_index = {cls: i for i, cls in enumerate(unique_classes)}

    train_labels_idx = torch.tensor(
        [class_to_index[label] for label in train_labels], dtype=torch.long
    )
    val_labels_idx = torch.tensor(
        [class_to_index[label] for label in val_labels], dtype=torch.long
    )

    num_classes = len(unique_classes)
    train_labels_onehot = nn.functional.one_hot(train_labels_idx, num_classes=num_classes).float()
    val_labels_onehot = nn.functional.one_hot(val_labels_idx, num_classes=num_classes).float()

    train_signals_np = np.stack(train_signals)
    train_data_tensor = torch.from_numpy(train_signals_np).unsqueeze(1).float()

    val_signals_np = np.stack(val_signals)
    val_data_tensor = torch.from_numpy(val_signals_np).unsqueeze(1).float()

    empty_t1 = torch.empty(train_data_tensor.shape[0], 1, train_data_tensor.shape[2])
    empty_t2 = torch.empty(train_data_tensor.shape[0], 1, train_data_tensor.shape[2])
    empty_t3 = torch.empty(val_data_tensor.shape[0], 1, val_data_tensor.shape[2])
    empty_t4 = torch.empty(val_data_tensor.shape[0], 1, val_data_tensor.shape[2])

    train_dataset = TensorDataset(torch.cat([train_data_tensor, empty_t1, empty_t2], dim=1), train_labels_onehot)
    validation_dataset = TensorDataset(torch.cat([val_data_tensor, empty_t3, empty_t4], dim=1), val_labels_onehot)

    logger.info('Train dataset size: {train_ds}.'.format(train_ds=train_dataset))
    logger.info('Validation dataset size: {validation_ds}.'.format(validation_ds=validation_dataset))

    return train_dataset, validation_dataset, num_classes


def prepare_data(replace_tensors_files: bool = False, signal_time: int | None = None) -> tuple[str, str, int, int]:
    """Import signals data, prepare dataset as tensor and save it to file."""

    training_data_set_file_path = os.getenv('TRAINING_DATA_SET_FILE_PATH')
    validation_data_set_file_path = os.getenv('VALIDATION_DATA_SET_FILE_PATH')

    if os.path.exists(training_data_set_file_path) and os.path.exists(validation_data_set_file_path):
        logger.info('Tensor files exists.')
        if not replace_tensors_files:
            logger.info('Skipping replacing tensors files.')
            num_classes = 3
            signal_length = 128  # TODO, make it accessible, read from ready tensor files
            return training_data_set_file_path, validation_data_set_file_path, num_classes, signal_length
        logger.info('Replacing existing tensor files with new ones.')

    logger.info('Generating tensor files.')

    mat_path, mat_file = download_dataset()

    mat_data = scipy.io.loadmat(mat_path)
    raw_data = mat_data[mat_file]['Data'][0, 0]
    num_signals = len(raw_data)
    len_signal = len(raw_data[0])
    logger.info('Gathered {num_signals} of length {len_signal} from {mat_file}'.format(
        num_signals=num_signals, len_signal=len_signal, mat_file=mat_file)
    )

    raw_signals_by_classes = split_signals_by_classes(mat_data[mat_file][0, 0])

    signals_of_desired_length, signal_length = split_signals_by_desired_length(raw_signals_by_classes, signal_time)

    signals_normalized = normalize_signals(signals_of_desired_length)

    train_data, train_labels, val_data, val_labels = create_train_and_validation_subsets(signals_normalized)

    train_dataset, val_dataset, num_classes = create_train_and_validation_tensors_datasets(
        train_data, train_labels, val_data, val_labels
    )

    train_tensors = train_dataset.tensors
    val_tensors = val_dataset.tensors

    torch.save(train_tensors, training_data_set_file_path)
    torch.save(val_tensors, validation_data_set_file_path)

    logger.info('Tensor files saved as {f1} and {f1}.'.format(
        f1=training_data_set_file_path, f2=validation_data_set_file_path)
    )

    return training_data_set_file_path, validation_data_set_file_path, num_classes, signal_length


if __name__ == '__main__':
    ex_training_data_file, ex_validation_data_file, ex_num_classes, ex_signal_length = prepare_data(signal_time=1)

    loaded_train_data, loaded_train_labels = torch.load(ex_training_data_file)
    loaded_val_data, loaded_val_labels = torch.load(ex_validation_data_file)

    train_ds = TensorDataset(loaded_train_data, loaded_train_labels)
    val_ds = TensorDataset(loaded_val_data, loaded_val_labels)

    batch_size = int(os.getenv('BATCH_SIZE'))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
