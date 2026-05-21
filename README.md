# RF Fingerprinting of Drone Remote Controllers Using Deep Learning

## Overview

This project investigates the use of deep learning for RF fingerprinting of drone remote controllers using transient radio-frequency emissions. The goal is to identify unique controller devices based solely on their transmitted RF waveform characteristics.

The work uses the MPACT DroneRC RF Dataset and focuses on learning discriminative transient features directly from raw RF waveforms without converting signals into spectrograms or image-based representations.

The initial implementation establishes a complete end-to-end RF fingerprinting pipeline including:

* RF waveform preprocessing
* Transient detection and extraction
* Dataset construction
* Deep learning model training
* Multi-class controller classification
* Performance evaluation

---

# Dataset

## Dataset Used

MPACT DroneRC RF Dataset

The dataset contains RF waveform captures from 17 drone remote controller classes, including DJI, Spektrum, Futaba, FlySky, Graupner, Turnigy, and others.

Each waveform is stored as a MATLAB `.mat` file containing:

* Raw ADC waveform data
* Sampling metadata
* Voltage scaling parameters

---

# Controller Classes

The dataset contains the following controller classes:

* DJI_Inspire1Pro
* DJI_Matrice100
* DJI_Matrice600_1
* DJI_Matrice600_2
* DJI_Phantom3
* DJI_Phantom4Pro_1
* DJI_Phantom4Pro_2
* FlySky_FST6
* Futaba_T8FG
* Graupner_MC32
* HobbyKing_HKT6A
* JetiDuplex_DC16
* Spektrum_DX5e
* Spektrum_DX6e
* Spektrum_DX6i
* Spektrum_JRX9303
* Turnigy_9X

Total number of classes: 17

---

# Signal Characteristics

Each RF waveform contains:

* 5,000,000 samples
* Sampling frequency: 20 GHz
* Signal duration: 250 microseconds

Signals are real-valued voltage waveforms reconstructed from ADC counts using metadata stored in the MATLAB files.

---

# Project Workflow

## 1. Dataset Exploration

The initial stage involved:

* Enumerating controller folders
* Counting waveform captures per class
* Inspecting MATLAB file structure
* Extracting waveform metadata
* Visualizing waveform distributions

The analysis confirmed:

* Approximately balanced class distributions
* Consistent waveform lengths
* Stable RF transient regions

---

# 2. RF Waveform Processing

Each `.mat` file was processed by:

1. Loading waveform metadata
2. Extracting raw ADC samples
3. Converting ADC values to voltage
4. Constructing normalized RF waveforms

The conversion used:

```python
signal = yorg + yinc * signal_adc
```

where:

* `yinc` = voltage increment
* `yorg` = voltage origin offset

---

# 3. Transient Detection

The project focused on startup transient regions because RF fingerprints are often strongest during transmitter startup behavior.

A transient detector was implemented using:

* Absolute signal envelope
* Smoothed energy estimation
* Dynamic thresholding
* Sustained energy rise detection

Transient search was restricted to a validated signal region to reduce computation.

---

# 4. Transient Window Extraction

For each waveform:

* A transient start index was detected
* A fixed-length waveform window was extracted
* The transient segment was normalized

Window size used:

```text
200,000 samples
```

Normalization applied:

```python
transient = transient - mean
transient = transient / std
```

---

# 5. Dataset Construction

The processed dataset contained:

* 850 waveform samples
* 17 controller classes
* 50 captures per class

Final tensor shape:

```text
X shape: (850, 200000)
y shape: (850,)
```

After channel expansion for Conv1D:

```text
X shape: (850, 200000, 1)
```

---

# 6. Train / Validation / Test Split

The dataset was split using stratified sampling:

* 80% Training
* 10% Validation
* 10% Testing

Final dataset sizes:

| Split      | Shape            |
| ---------- | ---------------- |
| Train      | (680, 200000, 1) |
| Validation | (85, 200000, 1)  |
| Test       | (85, 200000, 1)  |

---

# Deep Learning Model

## CNN Architecture

A 1D convolutional neural network was implemented for raw RF waveform classification.

The architecture included:

* Large temporal convolution kernels
* Aggressive early downsampling
* Multiple Conv1D feature extraction blocks
* Global average pooling
* Dense classifier head

---

# Model Structure

## Block 1

* Conv1D (64 filters, kernel size 31, stride 8)
* ReLU
* MaxPooling1D

## Block 2

* Conv1D (128 filters, kernel size 15)
* ReLU
* MaxPooling1D

## Block 3

* Conv1D (256 filters, kernel size 7)
* ReLU
* MaxPooling1D

## Block 4

* Conv1D (256 filters, kernel size 5)
* ReLU

## Classification Head

* GlobalAveragePooling1D
* Dense(256)
* Dropout(0.4)
* Softmax output layer

Total trainable parameters:

```text
752,785
```

---

# Training Configuration

## Optimizer

Adam optimizer:

```python
learning_rate = 3e-4
```

## Loss Function

```python
SparseCategoricalCrossentropy
```

## Callbacks

* EarlyStopping
* ReduceLROnPlateau

## Batch Size

```text
16
```

## Epochs

```text
50
```

---

# Experimental Results

## Final Test Performance

| Metric        | Value  |
| ------------- | ------ |
| Test Accuracy | 94.12% |
| Test Loss     | 0.2408 |

---

# Classification Performance

The model achieved strong performance across most controller classes.

Several classes achieved perfect classification performance:

* DJI_Inspire1Pro
* DJI_Matrice100
* DJI_Phantom3
* FlySky_FST6
* Futaba_T8FG
* JetiDuplex_DC16
* Spektrum_DX5e
* Spektrum_DX6e
* Spektrum_DX6i
* Spektrum_JRX9303
* Turnigy_9X

---

# Challenging Classes

The primary confusion occurred between:

* DJI_Matrice600_1 vs DJI_Matrice600_2
* DJI_Phantom4Pro_1 vs DJI_Phantom4Pro_2

These controllers likely share:

* Similar RF hardware
* Similar transmission protocols
* Similar oscillator characteristics

making fine-grained RF separation more difficult.

---

# Key Findings

## Successful Outcomes

The project demonstrated that:

* Raw RF transient waveforms contain identifiable device-specific information
* Deep learning can successfully classify drone controllers directly from waveform data
* Large temporal convolution kernels are effective for RF transient analysis
* Startup transient regions contain strong RF fingerprint information

---

# Limitations Identified

Several limitations were observed in the initial implementation:

## 1. Alignment Instability

Transient detection relied on threshold-based methods, resulting in temporal misalignment between captures.

## 2. Limited Dataset Usage

Only 50 captures per class were used despite larger available datasets.

## 3. No RF-Specific Augmentation

The model was trained without:

* AWGN injection
* temporal shifting
* phase augmentation
* frequency drift simulation

## 4. Pure CNN Architecture

The model relied solely on local convolutional operations and could not fully model long-range temporal dependencies.

---

# Future Work

Future development will focus on:

* Robust waveform alignment
* RF-specific augmentation strategies
* Embedding-based metric learning
* ArcFace classification
* Transformer-based sequence modeling
* Conformer architectures
* Self-supervised RF representation learning

The long-term objective is to push classification accuracy toward:

```text
99%+
```

using end-to-end raw waveform RF fingerprint learning.

---

# Technologies Used

* Python
* NumPy
* SciPy
* Pandas
* Matplotlib
* Seaborn
* TensorFlow / Keras
* Scikit-learn

---

# Research Direction

This work forms the baseline stage of a larger research effort focused on:

```text
End-to-End Raw RF Fingerprinting Using Transformer and Conformer Architectures
```

The future system aims to learn device identity directly from raw RF waveforms without spectrogram conversion or handcrafted feature engineering.
