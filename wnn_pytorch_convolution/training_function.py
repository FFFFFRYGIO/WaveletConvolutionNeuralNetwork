import torch
from sklearn.metrics import precision_score, confusion_matrix
from torch import nn

from wnn_create_network import WaveletNauralNet

SIGNAL_LENGTH = 128
num_classes = 3
batch_size = 32


def train_and_validate(model, train_loader, val_loader, epochs, lr, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_accs = []
    val_precisions = []
    val_confmats = []
    first_layers_params_monitor = []
    last_layer_params_monitor = []

    # model.to(device)

    for epoch in range(epochs):
        current_first_layer_params = model.conv_layers[0].weights.tolist()
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
            x1, x2, x3 = data[:, 0:1, :], data[:, 1:2, :], data[:, 2:3, :]

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
                x1, x2, x3 = data[:, 0:1, :], data[:, 1:2, :], data[:, 2:3, :]

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
        epoch_acc = (y_pred == y_true).mean()
        epoch_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        conf_mat = confusion_matrix(y_true, y_pred)

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        val_accs.append(epoch_acc)
        val_precisions.append(epoch_prec)
        val_confmats.append(conf_mat)

        print(f"Epoch {epoch + 1:03d}/{epochs:03d} — "
              f"Train loss: {epoch_train_loss:.4f}, "
              f"Val loss: {epoch_val_loss:.4f}, "
              f"Val Acc: {epoch_acc:.4f}, "
              f"Val Prec: {epoch_prec:.4f}, "
              f"Confusion Matrix: {conf_mat.tolist()}, "
              f"1st layer weights: {current_first_layer_params[0:4]}, "
              f"last layer weights: {current_last_layer_params[0][0:4]}, ")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'val_precisions': val_precisions,
        'val_confmats': val_confmats,
        'first_layers_params_monitor': first_layers_params_monitor,
        'last_layer_params_monitor': last_layer_params_monitor,
    }
