# -*- coding: utf-8 -*-
"""
Modality-Aware Graph Attention Network for In-Situ Composition Monitoring.

This script implements a Graph Neural Network with attention mechanisms for monitoring
material composition during Laser Powder Bed Fusion (LPBF) processes using fused
optical and acoustic emission sensor data.

The implementation includes:
- Data loading and preprocessing
- Graph construction from time series data
- Model training and evaluation
- Extensive visualization capabilities

@author: vpsora
contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com

Note:
    Any reuse of this code should be authorized by the code author.
    Developed for the publication:
    "Modality-Aware and Light-Weight Graph Attention Networkfor In-SituComposition Monitoring 
    in PBF-LB of Graded 316L–CuCrZr Alloys by Sensor Fusion of Optical and Acoustic Emissions"
"""
# %%
# Libraries to import
import os
import gc
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import DataLoader

from Config import SEED, data_folder, plot_folder, num_epochs, batch_size, test_size, shapelet_len, num_shapelets
from Network import GNNWithAttention, BatchedShapeletExtractor
from Trainer import train_model, test_model
from Utils import *
from Inference import *
from Visualization import *


# %%
def initialize_environment():
    """
    Initialize the computational environment with fixed random seeds and GPU settings.

    This function:
    - Sets random seeds for reproducibility
    - Configures PyTorch for deterministic operations
    - Clears GPU memory cache
    - Initializes GPU random seed if available
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gc.collect()
    torch.cuda.empty_cache()
    set_gpu_seed(SEED)


def load_and_preprocess_data():
    """
    Load and preprocess the sensor data for model training.

    Returns:
        tuple: (D1, D2, Y, class_labels, num_classes)
            - D1: Standardized optical emission data (numpy array)
            - D2: Standardized acoustic emission data (numpy array)
            - Y: Encoded class labels (numpy array)
            - class_labels: Original class labels
            - num_classes: Number of unique classes
    """
    # Load raw data
    D1_full = np.load(os.path.join(data_folder, "D1_rawspace_5000.npy"))
    D2_full = np.load(os.path.join(data_folder, "D2_rawspace_5000.npy"))
    Y_full = np.load(os.path.join(data_folder, "classspace_5000.npy"))

    # Class balancing
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

    # Encode labels
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    class_labels = label_encoder.classes_
    num_classes = len(class_labels)

    # Normalize data
    D1 = standardize(D1)
    D2 = standardize(D2)

    return D1, D2, Y, class_labels, num_classes, label_encoder


def create_graph_dataset(D1, D2, Y):
    """
    Convert time series data into graph representations.

    Args:
        D1: Optical emission time series data
        D2: Acoustic emission time series data
        Y: Corresponding class labels

    Returns:
        list: List of graph objects ready for PyTorch Geometric
    """
    print("[BUILD] Constructing graph representations from time series...")
    graph_list = []
    for i in range(len(D1)):
        graph = create_shapelet_graph_batched(D1[i], D2[i], Y[i])
        graph_list.append(graph)
    return graph_list


def initialize_model(num_classes, device):
    """
    Initialize the GNN model with attention mechanisms.

    Args:
        num_classes: Number of output classes
        device: Computation device ('cuda' or 'cpu')

    Returns:
        tuple: (model, optimizer, criterion)
            - Initialized GNN model
            - Adam optimizer
            - CrossEntropyLoss criterion
    """
    model = GNNWithAttention(
        in_channels=num_shapelets * 2,
        hidden_channels=16,
        out_channels=num_classes,
        shapelet_len=shapelet_len,
        num_shapelets=num_shapelets
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion


def run_visualizations(model, test_graphs, all_preds, all_labels, device, class_labels):
    """
    Execute all visualization functions for model interpretation.

    Args:
        model: Trained GNN model
        test_graphs: Test dataset graphs
        all_preds: Model predictions
        all_labels: Ground truth labels
        device: Computation device
        class_labels: List of class names
    """
    # Get correctly predicted examples
    correct_indices = (all_preds == all_labels).nonzero(as_tuple=True)[0]
    correct_graphs = [test_graphs[i] for i in correct_indices]
    correct_preds = [all_preds[i] for i in correct_indices]
    correct_labels = [all_labels[i] for i in correct_indices]

    # Shapelet visualizations
    ch1_init, ch2_init = model.get_shapelets_initial()
    ch1_trained, ch2_trained = model.get_shapelets_current()
    visualize_shapelets(ch1_init, ch2_init, plot_folder,
                        save_name="Initialized Shapelets")
    visualize_shapelets(ch1_trained, ch2_trained,
                        plot_folder, save_name="Updated Shapelets")
    plot_shapelets_side_by_side(
        ch1_init, ch1_trained, ch2_init, ch2_trained, plot_folder=plot_folder)

    # Node activation visualizations
    shapelet_model = model.shapelet_model
    visualize_classwise_node_activations_per_channel(correct_graphs, correct_labels,
                                                     shapelet_model, plot_folder, use_mean=False)
    visualize_classwise_node_activations_stacked(correct_graphs, correct_labels,
                                                 shapelet_model, plot_folder, use_mean=False)
    visualize_node_activations_from_multiple_graphs(correct_graphs, correct_preds, correct_labels,
                                                    shapelet_model, plot_folder, use_mean=False)

    # Attention visualizations
    plot_classwise_attention_maps(
        model, correct_graphs, correct_labels, device=device, save_folder=plot_folder)
    A_global = plot_global_attention_heatmap(model, correct_graphs, device, save_folder=plot_folder,
                                             normalize_rows=True, top_k=None, title="Global GAT Attention")

    # Embedding visualizations
    visualize_embeddings_all_methods(model=model, graph_list=correct_graphs, device=device,
                                     class_labels=class_labels, save_folder=plot_folder)

    # Saliency analysis
    summary = saliency_channel_contributions_per_class(model, correct_graphs, device,
                                                       plot_folder=plot_folder, save_prefix="Saliency")
    avg_saliency_per_class, counts = Node_saliency_per_class(model, correct_graphs, correct_labels, device,
                                                             plot_folder=plot_folder, save_prefix="Node-saliency")


def main():
    """Main execution function for the LPBF monitoring system."""
    # Initialize environment
    initialize_environment()
    plot_temporal_graph_representation()

    # Load and preprocess data
    D1, D2, Y, class_labels, num_classes, label_encoder = load_and_preprocess_data()
    plot_combined_series(D1, D2, Y, plot_folder=plot_folder)

    # Create graph dataset
    graph_list = create_graph_dataset(D1, D2, Y)
    train_graphs, test_graphs = train_test_split(
        graph_list, test_size=test_size, stratify=Y, random_state=42)
    graph_stats(train_graphs, "Train")
    graph_stats(test_graphs, "Test")

    # Create data loaders
    train_loader = DataLoader(
        train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # Initialize and train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, optimizer, criterion = initialize_model(num_classes, device)
    model_trained, model_save_path = train_model(model, train_loader, test_loader, optimizer,
                                                 criterion, device, class_labels)

    # Load best model and test
    model = GNNWithAttention(
        in_channels=num_shapelets * 2,
        hidden_channels=16,
        out_channels=num_classes,
        shapelet_len=shapelet_len,
        num_shapelets=num_shapelets
    ).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model_trained = model.to(device)

    # Evaluate model
    all_preds, all_labels = test_model(
        model_trained, test_loader, device, label_encoder)

    # Run visualizations
    run_visualizations(model_trained, test_graphs, all_preds,
                       all_labels, device, class_labels)
    print("All visualizations completed!")


if __name__ == "__main__":
    main()
