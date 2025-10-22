# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 11:59:21 2025

@author: vpsora
Any reuse of this code should be authorized by the code author.
Developed for the publication:
"Modality-Aware and Light-Weight Graph Attention Networkfor In-SituComposition Monitoring 
in PBF-LB of Graded 316L–CuCrZr Alloys by Sensor Fusion of Optical and Acoustic Emissions"
"""

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
SEED = 20  # Random seed for reproducibility


def set_gpu_seed(seed: int = SEED):
    """Set random seed for GPU operations to ensure reproducibility.

    Args:
        seed (int): Random seed value (default: 42)
    """
    try:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.ipc_collect()
    except:
        pass


def normalize(data):
    """Normalize data to [-1, 1] range"""
    print("[NORMALIZATION] Performing Min-Max normalization to [-1, 1]")
    data_min = np.min(data)
    data_max = np.max(data)
    print(f"[NORMALIZATION] Data min: {data_min:.4f}, max: {data_max:.4f}")

    normalized = 2 * ((data - data_min) / (data_max - data_min)) - 1

    norm_min = np.min(normalized)
    norm_max = np.max(normalized)
    print(f"[NORMALIZATION] Normalized data range: {
          norm_min:.4f} to {norm_max:.4f}")

    return normalized


def standardize(data):
    """Standardize data to zero mean and unit variance"""
    print("[STANDARDIZATION] Performing standardization")
    mean = np.mean(data)
    std = np.std(data)
    print(f"[STANDARDIZATION] Data mean: {mean:.4f}, std: {std:.4f}")

    standardized = (data - mean) / std

    data_min = np.min(standardized)
    data_max = np.max(standardized)
    print(f"[STANDARDIZATION] Standardized data range: {
          data_min:.4f} to {data_max:.4f}")

    return standardized


def count_parameters(model):
    """Count total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_curves(epochs, train_loss_means, train_loss_stds, val_accuracies, epoch_times, plot_folder):
    """
    Plots training loss (with std), validation accuracy, and epoch timing curves.

    Parameters:
    -----------
    epochs : list or np.ndarray
        List of epoch indices.
    train_loss_means : list or np.ndarray
        Mean training loss for each epoch.
    train_loss_stds : list or np.ndarray
        Std deviation of training loss for each epoch.
    val_accuracies : list or np.ndarray
        Validation accuracy per epoch.
    epoch_times : list or np.ndarray
        Duration of each epoch in seconds.
    plot_folder : str
        Folder to save plots. Created if it doesn't exist.
    """
    os.makedirs(plot_folder, exist_ok=True)

    # --- Training Loss with Std ---
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss_means, label='Train Loss', color='tab:blue')
    plt.fill_between(epochs,
                     np.array(train_loss_means) - np.array(train_loss_stds),
                     np.array(train_loss_means) + np.array(train_loss_stds),
                     alpha=0.3, color='tab:blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss with Std')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'train_loss_curve.png'))
    plt.show()

    # --- Validation Accuracy ---
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_accuracies, label='Val Accuracy', color='tab:green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'val_accuracy_curve.png'))
    plt.show()

    # --- Epoch Timing ---
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, epoch_times, marker='o', color='tab:orange')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Time per Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'epoch_time_curve.png'))
    plt.show()


def evaluate_gnn_model(gnn_model, test_loader, model_save_path, device, label_encoder, plot_folder):
    """
    Evaluates a trained GNN model on the test set.

    Parameters:
    -----------
    gnn_model : nn.Module
        Trained GNN model.
    test_loader : DataLoader
        DataLoader for the test set.
    model_save_path : str
        Path to the saved model checkpoint (.pt).
    device : torch.device
        Device to run evaluation on.
    label_encoder : LabelEncoder
        Fitted label encoder to decode class names.
    plot_folder : str
        Directory to save the confusion matrix plot.

    Returns:
    --------
    accuracy : float
        Overall test accuracy.
    avg_infer_time : float
        Average inference time per batch in seconds.
    """

    print("🚀 Evaluating on test set...")
    gnn_model.load_state_dict(torch.load(model_save_path))
    gnn_model.eval()

    all_preds, all_labels = [], []
    infer_times = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            t0 = time.time()
            out = gnn_model(batch.x, batch.edge_index, batch.batch)
            infer_times.append(time.time() - t0)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(batch.y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    print("\n=== ✅ Test Results ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"✅ Overall Accuracy: {accuracy * 100:.2f}%")

    # Confusion matrix (normalized without % symbol)
    cm_normalized = confusion_matrix(
        all_labels, all_preds, normalize='true') * 100

    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, values_format=".1f")

    plt.title("Confusion Matrix (normalized by row)")
    plt.tight_layout()
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder, "confusion_matrix_percent.png"))
    plt.show()

    avg_infer_time = np.mean(infer_times)
    print(f"📊 Avg. Inference Time per Batch: {avg_infer_time:.4f} sec")

    return accuracy, avg_infer_time


def create_shapelet_graph_batched(d1_sample, label, window_size=500, stride=250):
    """
    Constructs a graph from single-channel time series using sliding windows.

    Parameters:
    -----------
    d1_sample : np.ndarray
        1D numpy array of shape (T,) representing the time-series signal.
    label : int
        Integer class label for the time series.
    window_size : int, optional
        Size of the sliding window (default is 500).
    stride : int, optional
        Stride length between consecutive windows (default is 250).

    Returns:
    --------
    Data : torch_geometric.data.Data
        A graph where each node corresponds to a windowed segment (1×window_size),
        and edges connect temporally adjacent segments.
    """
    N = d1_sample.shape[0]
    # print(f"Input d1_sample shape: {d1_sample.shape}")

    windows = []

    for i in range(0, N - window_size + 1, stride):
        d1_window = d1_sample[i:i + window_size]
        window = np.expand_dims(d1_window, axis=0)  # Shape: (1, window_size)
        windows.append(window)

    # Shape: (num_nodes, 1, window_size)
    # [num_nodes, 2, window_size]
    window_tensor = torch.tensor(windows, dtype=torch.float)
    num_nodes = len(windows)

    # Create fully connected bidirectional edges
    edge_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_list.append([i, j])

    edge_index = torch.tensor(
        edge_list, dtype=torch.long).t().contiguous()  # [2, num_edges]
    y = torch.tensor(label, dtype=torch.long)

    return Data(x=window_tensor, edge_index=edge_index, y=y)


def graph_stats(graphs, name=""):
    num_nodes = [g.num_nodes for g in graphs]
    print(f"📊 {name} Graph Stats")
    print(f"  - Total graphs: {len(graphs)}")
    print(f"  - Nodes per graph: min={min(num_nodes)
                                      }, max={max(num_nodes)}, avg={np.mean(num_nodes):.2f}")
