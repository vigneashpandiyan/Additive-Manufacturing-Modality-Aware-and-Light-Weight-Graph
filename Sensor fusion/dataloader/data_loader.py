# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Loading D1 (optical) and D2 (acoustic) waveforms and aligning them with alloy target class labels.
- Structuring multi-modal time series segments into structured PyTorch Geometric Data graph sequences.

Note: Any reuse of this code should be authorized by the code author.
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
    Description:
        Loads raw D1/D2 and class label `.npy` files, balances instances per composition class using resampling, encodes target classes, and standardizes sensor data.
    Purpose:
        To form a clean, balanced, and normalized pre-processed dataset for benchmark models.
    Input Types:
        - None
    Output Types:
        - D1 (numpy.ndarray): Balanced and normalized D1 sensor data.
        - D2 (numpy.ndarray): Balanced and normalized D2 sensor data.
        - Y (numpy.ndarray): Balanced and encoded target composition classes.
        - class_labels (numpy.ndarray): Mapping of encoded targets back to original labels.
        - num_classes (int): Count of unique composition classes.
        - label_encoder (sklearn.preprocessing.LabelEncoder): Encoder model used to map labels.
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
    Description:
        Translates raw multi-modal time series arrays into sets of windowed node graphs and stratifiably splits them into training and testing collections.
    Purpose:
        To construct graph representations necessary for evaluating proposed shapelet-GAT and baseline GNN architectures.
    Input Types:
        - D1 (numpy.ndarray): Raw D1 sensor matrix.
        - D2 (numpy.ndarray): Raw D2 sensor matrix.
        - Y (numpy.ndarray): Label target array.
    Output Types:
        - graph_list (list): training and evaluation graph dataset list.
    """
    print("[BUILD] Converting time-series samples to bidirectional temporal graphs...")
    graph_list = []
    for i in range(len(D1)):
        graph = create_shapelet_graph_batched(D1[i], D2[i], Y[i])
        graph_list.append(graph)
    return graph_list
