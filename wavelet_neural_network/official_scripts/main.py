"""Execute data generation, create WNN, run training and gather results."""
import os
from datetime import datetime

import matplotlib
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary

from prepare_data import prepare_data
from training_function import train_and_validate

from wnn_network_module.wavelet_neural_network import WaveletNeuralNet

from logger import logger


def get_data_loaders() -> tuple[DataLoader, DataLoader, int, int]:
    """Get data loaders based on prepare data script."""
    training_data_file, validation_data_file, num_classes, signal_length = prepare_data(signal_time=1)

    loaded_train_data, loaded_train_labels = torch.load(training_data_file)
    loaded_val_data, loaded_val_labels = torch.load(validation_data_file)

    train_ds = TensorDataset(loaded_train_data, loaded_train_labels)
    val_ds = TensorDataset(loaded_val_data, loaded_val_labels)

    batch_size = int(os.getenv('BATCH_SIZE'))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, num_classes, signal_length


def get_wnn_model(
        num_classes: int, signal_length: int, wavelet_name: str = 'db4', num_dense_layers: int = 3
) -> WaveletNeuralNet:
    """Create WNN model based on WaveletNeuralNet class."""
    wnn_model = WaveletNeuralNet(
        num_classes=num_classes,
        signal_length=signal_length,
        wavelet_name=wavelet_name,
        num_dense_layers=num_dense_layers,
    )
    return wnn_model


def plot_results(results: dict, save_to_file: bool = True, show_plots: bool = False) -> None:
    """Plot results from a training and validation process."""

    matplotlib.use("TkAgg")

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(10, 4))

    axes[0, 0].plot(results['train_losses'])
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(results['val_losses'])
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")

    axes[1, 0].plot(results['val_accuracies'])
    axes[1, 0].set_title("Validation Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")

    axes[1, 1].plot(results['val_precisions'])
    axes[1, 1].set_title("Validation Precision (macro)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Precision")

    plt.tight_layout()

    if save_to_file:
        save_path = f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(save_path)
    if show_plots:
        plt.show()


def main():
    """Execute data generation, create WNN, run training and gather results."""
    train_loader, val_loader, num_classes, signal_length = get_data_loaders()

    wnn_model = get_wnn_model(num_classes=num_classes, signal_length=signal_length)

    logger.info(wnn_model)

    one_batch_data, _ = next(iter(train_loader))

    x1_summary = one_batch_data[:, 0:1, :]
    x2_summary = one_batch_data[:, 1:2, :]
    x3_summary = one_batch_data[:, 2:3, :]

    summary(wnn_model, input_data=(x1_summary, x2_summary, x3_summary),
            col_names=["input_size", "output_size", "num_params", "trainable"])

    results = train_and_validate(
        model=wnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer_name='Adam',
        epochs=10,
        learning_rate=1e-3,
    )

    plot_results(results)


if __name__ == '__main__':
    main()
