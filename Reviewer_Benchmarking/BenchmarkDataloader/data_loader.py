# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities for Reviewer Benchmarking.
Reuses the dataset, standardization, and graph construction logic from the main repo.
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
BENCHMARK_DIR = os.path.dirname(CURRENT_DIR)             # Reviewer_Benchmarking/
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)            # Project Root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torch_geometric.data import Data

def standardize(data: np.ndarray) -> np.ndarray:
    """Standardize an array to zero mean and unit variance."""
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
    Construct a fully-connected bidirectional graph from two-channel time series
    via sliding windows.
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
    Load raw sensor data and perform class balancing and standardization.
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
    Format data for sequence models. Input format: [B, 2, 5000]
    """
    X = np.stack([D1, D2], axis=1) # Shape: [B, 2, 5000]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_split, stratify=Y, random_state=seed
    )
    
    return X_train, X_test, y_train, y_test


def get_graph_datasets(D1, D2, Y, test_split=test_size, seed=SEED):
    """
    Construct graph representation for graph models.
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
