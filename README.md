# Wavelet Convolution Neural Network

Python implementation attempt of data analysis and creating Wavelet Neural Network.


## Purpose of the Project

The main goal of this project is to run complex analysis of ECG signals using Wavelet Transform mathematical tool.
Based on the knowledge, the project contains implementation of Convolution Neural Network.
This network is dedicated to detect common cardiac condisions based on given signal.


## Project Theory

### ECG signal

ECG signal is the result of the electrocardiography process of detecting and recording the heart’s electrical activity
via surface electrodes, yielding a waveform that reflects cardiac depolarization and repolarization.

### Wavelet Transform

Wavelet Transform is a mathematical technique used to analyze one or multidimensional data. Its key part is a
Wavelet Function. After applying it mathematically to signal it results with decomposition coefficients.
This process is called time-frequency analysis.
Here for this project is used Discrete Wavelet Transform (DWT) that every decomposition level splits
the input into approximation and details coefficients.

### Convolution Neural Network

Convolution Neural Network is a version of neural networks designed to extract features from grid-like matrix datasets.
Layers of this network run convolution operations that apply learnable filters across the input to produce feature maps,
capturing hierarchical patterns in the data.

## Data analysis key steps

1. Samples distribution in ECG signal
2. Detection of the baseline of ECG signal
3. Normalization of one ECG signal
4. Denoising ECG signal
5. Wavelet functions efficiency comparison for ECG signal analysis
6. Number of wavelet transform decomposition levels efficiency analysis

## Architecture of Wavelet Neural Network

The Wavelet Neural Network was implemented using this list of layers:
1. Convolution layers that run DWT operations.
2. Parsing layer that converts convolution result into one flattened dataset.
3. Linear layers are used to classify convolution results.

The network’s output is a set of scores indicating how well the model’s signal representations match each class.

## Tech Stack and Development Environment

### Technology Stack

1. Python programming language.
2. PyTorch Python library for neural networks.
3. Torchinfo Python library for showing details about a created neural network model
4. PyWavelets Python library for wavelet transform.
5. pytorch_wavelets module to run wavelet transform on tensors.
6. SciPy Python library for data processing algorithms.

## Outcome of the project

The current outcome of this project is that it does not learn convolution parameters.
This happened because I was not able to implement a convolution layer class
that properly combines weights with data tensor.
Here is the description of the attempts that I have made to achieve that

### 1. Detach (script: `convolution_dwt_layer_detach.py`)

Usage of `.detach().numpy()` instruction to get dataset from tensor to run PyWavelets DWT operation on it.
Detaching the tensor stops gradient tracking and does not allow the weight to adjust.

### 2. Listing (script: `convolution_dwt_layer_listing.py`)

Usage of `.tolist()` instruction does not detach the tensor, but only getting the values from it.
The problem still persists because the weights are not combined with the tensor, only with its list representation.

### 3. pytorch_wavelets (script: `convolution_dwt_layer_pytorch_wavelets.py`)

`pytorch_wavelets` allows to run DWT operations on tensors.
Unfortunately, it does not allow parsing the weights as a tensor to wavelet scaling filter and still can't combine them.

### 4. convolution operation (script: `pytorch_convolution_dwt_test.py`)

That one contains an attempt to recreate wavelet transform operation using `torch.nn.functional.conv1d` instruction.
The script is returning two decompositions of the signal, but the result differs with expected values from PyWavelets.

## Project setup

1. Install requirements

`pip install -r requirements.txt`

2. Download and extract ECGData dataset from a zip file with ECG signals from this link:

`https://github.com/mathworks/physionet_ECG_data/raw/main/ECGData.zip`

3. Setup environment, here are the example values:

```
DATA_SOURCE=''  # Path to ECGData file
ECGDATA_FREQUENCY=128

TRAINING_DATA_SET_FILE_PATH='data/train_tensors.pt'
VALIDATION_DATA_SET_FILE_PATH='data/validation_tensors.pt'
DATA_SOURCE_URL='https://github.com/mathworks/physionet_ECG_data/raw/main/ECGData.zip'
ECG_DATA_SOURCE_SIGNAL_FREQUENCY=128
TRAIN_FRAC=0.8
BATCH_SIZE=32
```

4. Apply one of convolution DWT layer implementations

In `wavelet_neural_network/official_scripts/wnn_network_module`,
insert to `convolution_dwt_layer.py` script content of a chosen script from `conv_dwt_layers_variances`

## Author

Radosław Relidzyński

A student at the Military University of Technology

This project was created for a Master's of Science diploma.
