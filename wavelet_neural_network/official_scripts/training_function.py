"""Training and model validation implementation"""
import numpy as np
import torch
from sklearn.metrics import precision_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from wnn_network_module.wavelet_neural_network import WaveletNeuralNet

from logger import logger


def train_and_validate(
        model: WaveletNeuralNet,
        train_loader: DataLoader, val_loader: DataLoader,
        criterion: nn.modules.loss, optimizer_name: str,
        epochs: int, learning_rate: float, device: str = 'cpu'):
    """Run training and validation on a neural network."""
    logger.info('Starting training, configuration:')
    model_layers_num = len(model.conv_layers) + 1 + len(model.dense_layers)
    logger.info('Model layers: {layer_num}'.format(layer_num=model_layers_num))
    logger.info('Training data loader size: {train_size}'.format(train_size=len(train_loader)))
    logger.info('Validation data loader size: {val_size}'.format(val_size=len(val_loader)))
    logger.info('Loss function: {loss_function}'.format(loss_function=criterion))
    logger.info('Optimizer: {optimizer}'.format(optimizer=optimizer_name))
    logger.info('Epochs: {epochs}'.format(epochs=epochs))
    logger.info('Learning rate: {lr}'.format(lr=learning_rate))
    logger.info('Device: {device}'.format(device=device))

    try:
        optimizer_class = getattr(torch.optim, optimizer_name)
    except AttributeError:
        raise ValueError(f"Optimizer '{optimizer_name}' not found in torch.optim")

    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_conf_mats = []
    first_layers_params_monitor = []
    last_layer_params_monitor = []

    # model.to(device)

    for epoch in range(epochs):
        current_first_layer_params = model.conv_layers[0].weights.tolist()
        first_layers_params_monitor.append(current_first_layer_params)
        try:
            current_last_layer_params = model.dense_layers[-1].weight.cpu().detach().numpy()
        except AttributeError as e:
            logger.info('Skipping,', e)
            current_last_layer_params = [None] * len(current_first_layer_params)
        last_layer_params_monitor.append(current_first_layer_params)

        # — TRAIN —
        model.train()
        running_train_loss = 0.0
        for data, target in train_loader:
            # data, target = data.to(device), target.to(device)
            # split channels if needed
            x1, x2, x3 = data[:, 0:1, :], data[:, 1:2, :], data[:, 2:3, :]

            optimizer.zero_grad()
            train_outputs = model(x1, x2, x3)
            train_loss = criterion(train_outputs, target.argmax(dim=1))
            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # — VALIDATE —
        model.eval()
        running_val_loss = 0.0
        all_predictions = []
        all_trues = []
        with torch.no_grad():
            for data, target in val_loader:
                # data, target = data.to(device), target.to(device)
                x1, x2, x3 = data[:, 0:1, :], data[:, 1:2, :], data[:, 2:3, :]

                val_outputs = model(x1, x2, x3)
                val_loss = criterion(val_outputs, target.argmax(dim=1))

                predictions = val_outputs.argmax(dim=1)

                running_val_loss += val_loss.item()

                all_predictions.append(predictions.cpu())
                all_trues.append(target.argmax(dim=1).cpu())

        # concatenate across batches
        y_pred = torch.cat(all_predictions).numpy()
        y_true = torch.cat(all_trues).numpy()

        # accuracy, precision, confusion matrix
        epoch_acc = np.mean(y_pred == y_true)
        epoch_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        conf_mat = confusion_matrix(y_true, y_pred)

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        val_accuracies.append(epoch_acc)
        val_precisions.append(epoch_precision)
        val_conf_mats.append(conf_mat)

        logger.info(f"Epoch {epoch + 1:03d}/{epochs:03d} — "
                    f"Train loss: {epoch_train_loss:.4f}, "
                    f"Val loss: {epoch_val_loss:.4f}, "
                    f"Val Acc: {epoch_acc:.4f}, "
                    f"Val Precision: {epoch_precision:.4f}, "
                    f"Confusion Matrix: {conf_mat.tolist()}, "
                    f"1st layer weights: {current_first_layer_params[0:4]}, "
                    f"last layer weights: {current_last_layer_params[0][0:4]}, ")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_precisions': val_precisions,
        'val_conf_mats': val_conf_mats,
        'first_layers_params_monitor': first_layers_params_monitor,
        'last_layer_params_monitor': last_layer_params_monitor,
    }
