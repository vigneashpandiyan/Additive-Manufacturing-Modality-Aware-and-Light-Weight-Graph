# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Loading and parsing raw sensor inputs and composition label indices.
- Constructing sequence vectors and batched datasets for 1D CNN, RNN, and TCN models.

Note: Any reuse of this code should be authorized by the code author.
"""

import os
import sys
import numpy as np
import torch
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Resolve paths dynamically relative to subfolder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # dataloader/
BENCHMARK_DIR = os.path.dirname(CURRENT_DIR)             # Model Benchmarking/
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)            # Project Root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torch_geometric.data import Data

def standardize(data: np.ndarray) -> np.ndarray:
    """
    Description:
        Standardizes the input sensor signal array to zero mean and unit variance.
    Purpose:
        To stabilize input features across different modalities and support convergent gradient updates.
    Input Types:
        - data (numpy.ndarray): Multi-dimensional raw sensor data array.
    Output Types:
        - standardized (numpy.ndarray): Standardized sensor data array.
    """
    print("[STANDARDIZATION] Performing standardization")
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        std = 1e-12
    standardized = (data - mean) / std
    return standardized

def create_shapelet_graph_batched(
    d1_sample: np.ndarray,
    d2_sample: np.ndarray,
    label: int,
    window_size: int = 500,
    stride: int = 250,
) -> Data:
    """
    Description:
        Partitions two-channel signals using overlapping sliding windows to create a bidirectional, fully-connected Graph (PyTorch Geometric Data).
    Purpose:
        To map spatial-temporal waveform relations to nodes and edges in a graph representation.
    Input Types:
        - d1_sample (numpy.ndarray): 1D array of optical sensor signal values.
        - d2_sample (numpy.ndarray): 1D array of acoustic sensor signal values.
        - label (int): Alloy composition class index target.
        - window_size (int): Temporal size of each window. Default is 500.
        - stride (int): Overlap shift distance. Default is 250.
    Output Types:
        - Data (torch_geometric.data.Data): Graph containing window node tensors and edge index maps.
    """
    assert d1_sample.ndim == 1 and d2_sample.ndim == 1, "Inputs must be 1D arrays."
    assert d1_sample.shape[0] == d2_sample.shape[0], "Channel lengths must match."
    assert window_size > 0 and stride > 0, "window_size and stride must be positive."

    T = d1_sample.shape[0]
    windows = []

    for start in range(0, T - window_size + 1, stride):
        d1_window = d1_sample[start:start + window_size]
        d2_window = d2_sample[start:start + window_size]
        window = np.stack([d1_window, d2_window], axis=0)  # [2, window_size]
        windows.append(window)

    if len(windows) == 0:
        raise ValueError("No windows were created. Decrease window_size or stride.")

    window_tensor = torch.tensor(np.stack(windows, axis=0), dtype=torch.float)
    num_nodes = window_tensor.shape[0]

    edge_src = []
    edge_dst = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_src.append(i)
                edge_dst.append(j)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=window_tensor.device)

    y = torch.tensor(label, dtype=torch.long)

    return Data(x=window_tensor, edge_index=edge_index, y=y)

# Default splits configuration
SEED = 42
test_size = 0.30

DATA_FOLDER = os.path.join(PROJECT_ROOT, "Data")


def load_raw_data(seed=SEED):
    """
    Description:
        Loads raw D1/D2 and class label `.npy` files, balances instances per composition class using resampling, encodes target classes, and standardizes sensor data.
    Purpose:
        To form a clean, balanced, and normalized pre-processed dataset for benchmark models.
    Input Types:
        - seed (int): Seed number to initialize resampling and establish reproducibility.
    Output Types:
        - D1 (numpy.ndarray): Resampled and normalized D1 sensor data.
        - D2 (numpy.ndarray): Resampled and normalized D2 sensor data.
        - Y (numpy.ndarray): Balanced and encoded target composition classes.
        - class_labels (numpy.ndarray): Mapping of encoded targets back to original labels.
        - num_classes (int): Count of unique composition classes.
        - label_encoder (sklearn.preprocessing.LabelEncoder): Encoder model used to map labels.
    """
    print("[DATA] Loading raw data...")
    D1_full = np.load(os.path.join(DATA_FOLDER, "D1_rawspace_5000.npy"))
    D2_full = np.load(os.path.join(DATA_FOLDER, "D2_rawspace_5000.npy"))
    Y_full = np.load(os.path.join(DATA_FOLDER, "classspace_5000.npy"))

    print(f"[DATA] Raw shapes: D1={D1_full.shape}, D2={D2_full.shape}, Y={Y_full.shape}")

    # Class balancing
    samples_per_class = int(len(Y_full) / len(np.unique(Y_full)))
    selected_indices = []
    min_count = min(np.sum(Y_full == c) for c in np.unique(Y_full))

    for cls in np.unique(Y_full):
        idx = np.where(Y_full == cls)[0]
        n = min(samples_per_class, len(idx), min_count)
        selected = resample(idx, replace=True, n_samples=n, random_state=seed)
        selected_indices.extend(selected)

    D1 = D1_full[selected_indices]
    D2 = D2_full[selected_indices]
    Y = Y_full[selected_indices]

    # Encode labels
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    class_labels = label_encoder.classes_
    num_classes = len(class_labels)

    # Normalize/Standardize
    D1 = standardize(D1)
    D2 = standardize(D2)

    print(f"[DATA] Balanced shapes: D1={D1.shape}, D2={D2.shape}, Y={Y.shape}")
    print(f"[DATA] Number of classes: {num_classes} ({class_labels})")

    return D1, D2, Y, class_labels, num_classes, label_encoder


def get_sequence_datasets(D1, D2, Y, test_split=test_size, seed=SEED):
    """
    Description:
        Packages D1 and D2 arrays into sequence representations of size [Batch, Channels, Timesteps] and splits them stratifiably into training and testing sets.
    Purpose:
        To format time-series inputs correctly for 1D CNN, LSTM, TCN, and Transformer sequence baselines.
    Input Types:
        - D1 (numpy.ndarray): Normalised D1 sensor matrix.
        - D2 (numpy.ndarray): Normalised D2 sensor matrix.
        - Y (numpy.ndarray): Encoded label array.
        - test_split (float): Ratio of testing subset.
        - seed (int): Random seed to control data splits.
    Output Types:
        - X_train (numpy.ndarray): Structured train sequence inputs.
        - X_test (numpy.ndarray): Structured test sequence inputs.
        - y_train (numpy.ndarray): Train target sequence labels.
        - y_test (numpy.ndarray): Test target sequence labels.
    """
    X = np.stack([D1, D2], axis=1) # Shape: [B, 2, 5000]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_split, stratify=Y, random_state=seed
    )
    
    return X_train, X_test, y_train, y_test


def get_graph_datasets(D1, D2, Y, test_split=test_size, seed=SEED):
    """
    Description:
        Translates raw multi-modal time series arrays into sets of windowed node graphs and stratifiably splits them into training and testing collections.
    Purpose:
        To construct graph representations necessary for evaluating proposed shapelet-GAT and baseline GNN architectures.
    Input Types:
        - D1 (numpy.ndarray): Raw D1 sensor matrix.
        - D2 (numpy.ndarray): Raw D2 sensor matrix.
        - Y (numpy.ndarray): Label target array.
        - test_split (float): Proportion of the test split.
        - seed (int): Seed number used for stratified random splits.
    Output Types:
        - train_graphs (list): Training collection of window graphs.
        - test_graphs (list): Test collection of window graphs.
    """
    print("[DATA] Building graph dataset from time series (this may take a minute)...")
    graph_list = []
    for i in range(len(D1)):
        graph = create_shapelet_graph_batched(D1[i], D2[i], Y[i])
        graph_list.append(graph)
        
    train_graphs, test_graphs = train_test_split(
        graph_list, test_size=test_split, stratify=Y, random_state=seed
    )
    
    return train_graphs, test_graphs
