# Time Series Anomaly Injection Library
This library provides Python tools for injecting various types of synthetic anomalies into multivariate time series data. It can be utilized for data augmentation in anomaly detection model training, robustness testing, or generating evaluation datasets for anomaly detection algorithms.

# Overview
This library offers flexible functionalities to inject diverse anomaly types such as `spike, flip, speedup, noise, cutoff, moving average, scale change, wander, contextual, upsidedown, and mixture` into existing time series datasets. It supports both a mode for processing entire datasets and a mode for handling pre-windowed batch data. Additionally, it includes an optional feature to apply Min-Max scaling and revert the data to its original scale after processing.

# Features
- `Diverse Anomaly Types:` Supports over 10 different anomaly patterns.
- `Flexible Parameter Settings:` Allows detailed customization of parameters (e.g., scale, range, number of features) for each anomaly type.
- `Dataset-wide Injection:` Uses DataLoaderAug to extract windows from an entire time series dataset and inject anomalies.
- `Batch Data Injection:` Uses DataLoaderAugBatch to inject anomalies into already windowed PyTorch tensor data.
- `Min-Max Scaling:` An optional feature to scale data to a 0-1 range before anomaly injection and revert it to the original scale after processing.
- `Anomaly Mask and Labels:` Generates a mask indicating the exact locations where anomalies were injected and One-Hot encoded labels representing the anomaly type for each window.
- `Random Anomaly Injection:` By specifying the 'random' type, anomalies are randomly selected from predefined types for injection.

# Installation
This library is written in Python and can be installed as below.
```
pip install
```
