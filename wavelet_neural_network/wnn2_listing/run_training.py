import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from training_function import train_and_validate
from wnn_create_network import WaveletNauralNet

matplotlib.use("TkAgg")

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

model = WaveletNauralNet(num_classes=num_classes)

print('STARTING WEIGHTS')
for layer in model.conv_layers:
    print(layer.weights.tolist())

# Example usage:
results = train_and_validate(
    model, train_loader, val_loader,
    epochs=10, lr=1e-3
)

print('FINISH WEIGHTS', model.conv_layers[0].weights.tolist())
for layer in model.conv_layers:
    print(layer.weights.tolist())

# Plotting Loss, Accuracy, Precision

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

axes[1, 0].plot(results['val_accs'])
axes[1, 0].set_title("Validation Accuracy")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Accuracy")

axes[1, 1].plot(results['val_precisions'])
axes[1, 1].set_title("Validation Precision (macro)")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Precision")

# first_layer_params_means = [np.mean(params) for params in results['first_layers_params_monitor']]
# axes[2, 0].plot(first_layer_params_means)
# axes[2, 0].set_title("Mean values of first layer parameters")
# axes[2, 0].set_xlabel("Epoch")
# axes[2, 0].set_ylabel("Mean value")
#
# first_layer_params_stds = [np.std(params) for params in results['first_layers_params_monitor']]
# axes[2, 1].plot(first_layer_params_stds)
# axes[2, 1].set_title("Standard deviation of first layer parameters")
# axes[2, 1].set_xlabel("Epoch")
# axes[2, 1].set_ylabel("Std value")
#
# last_layer_params_means = [np.mean(params) for params in results['last_layer_params_monitor']]
# axes[3, 0].plot(last_layer_params_means)
# axes[3, 0].set_title("Mean values of last layer parameters")
# axes[3, 0].set_xlabel("Epoch")
# axes[3, 0].set_ylabel("Mean value")
#
# last_layer_params_stds = [np.std(params) for params in results['last_layer_params_monitor']]
# axes[3, 1].plot(last_layer_params_stds)
# axes[3, 1].set_title("Standard deviation of last layer parameters")
# axes[3, 1].set_xlabel("Epoch")
# axes[3, 1].set_ylabel("Std value")

plt.tight_layout()
plt.show()

# To inspect the final confusion matrix:
print("Final Confusion Matrix:")
print(results['val_confmats'][-1])
