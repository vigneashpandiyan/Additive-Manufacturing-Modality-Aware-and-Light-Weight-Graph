# -*- coding: utf-8 -*-
"""
Dataloader and preprocessing module for dual-channel sensor signals.
"""

import os
import sys
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

# Resolve parent directory to allow imports of Config and Utils
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Config import SEED, data_folder
from utils.Utils import standardize, create_shapelet_graph_batched


def load_and_preprocess_data():
    """
    Load raw sensor signals, perform stratified resampling for class balancing,
    standardize each sensor modality independently, and encode labels.
    
    Returns:
        tuple: (D1, D2, Y, class_labels, num_classes, label_encoder)
    """
    print("[DATA] Loading raw datasets...")
    D1_full = np.load(os.path.join(data_folder, "D1_rawspace_5000.npy"))
    D2_full = np.load(os.path.join(data_folder, "D2_rawspace_5000.npy"))
    Y_full = np.load(os.path.join(data_folder, "classspace_5000.npy"))

    print(f"[DATA] Original dataset size: {len(Y_full)} samples")

    # Class balancing (stratified resampling)
    samples_per_class = int(1 * len(Y_full) / len(np.unique(Y_full)))
    selected_indices = []
    min_count = min(np.sum(Y_full == c) for c in np.unique(Y_full))

    for cls in np.unique(Y_full):
        idx = np.where(Y_full == cls)[0]
        n = min(samples_per_class, len(idx), min_count)
        selected = resample(idx, replace=True, n_samples=n, random_state=SEED)
        selected_indices.extend(selected)

    D1 = D1_full[selected_indices]
    D2 = D2_full[selected_indices]
    Y = Y_full[selected_indices]

    print(f"[DATA] Balanced dataset size: {len(Y)} samples")

    # Encode labels
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    class_labels = label_encoder.classes_
    num_classes = len(class_labels)

    # Standardize data
    D1 = standardize(D1)
    D2 = standardize(D2)

    return D1, D2, Y, class_labels, num_classes, label_encoder


def create_graph_dataset(D1, D2, Y):
    """
    Convert raw dual-channel series samples into PyG bidirectional temporal graphs.
    """
    print("[BUILD] Converting time-series samples to bidirectional temporal graphs...")
    graph_list = []
    for i in range(len(D1)):
        graph = create_shapelet_graph_batched(D1[i], D2[i], Y[i])
        graph_list.append(graph)
    return graph_list
