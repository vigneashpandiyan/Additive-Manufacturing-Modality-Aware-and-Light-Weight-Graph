# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Normalizations, reproducibility settings, and unimodal shapelet-graph extraction helpers.
- Model performance evaluation and diagnostic training curves visualization functions.

Note: Any reuse of this code should be authorized by the code author.
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
    """
    Description:
        Sets the random seed for GPU operations (specifically CUDA) and forces garbage collection.
    Purpose:
        To guarantee reproducibility of deep learning calculations across GPU/CUDA executions.
    Input Types:
        - seed (int): The integer random seed value to enforce.
    Output Types:
        - None
    """
    try:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.ipc_collect()
    except:
        pass


def normalize(data):
    """
    Description:
        Normalizes the input numpy array to [-1, 1] range using Min-Max scaling.
    Purpose:
        To scale input signals uniformly to a balanced dynamic range.
    Input Types:
        - data (np.ndarray): Numeric array/matrix representing raw signal segments.
    Output Types:
        - normalized (np.ndarray): Scaled numeric array in the range [-1, 1].
    """
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
    """
    Description:
        Standardizes the input numpy array to zero mean and unit variance.
    Purpose:
        To scale signal features by centered mean and standard deviation.
    Input Types:
        - data (np.ndarray): Numeric array representing raw signals.
    Output Types:
        - standardized (np.ndarray): Zero-mean, unit-variance standardized array.
    """
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
    """
    Description:
        Counts the total number of trainable (requires_grad=True) parameters in the given PyTorch model.
    Purpose:
        To calculate model complexity and scale footprint measurements for benchmarking.
    Input Types:
        - model (nn.Module): PyTorch neural network model.
    Output Types:
        - total_params (int): Total count of trainable weights/parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_curves(epochs, train_loss_means, train_loss_stds, val_accuracies, epoch_times, plot_folder):
    """
    Description:
        Plots and exports visual diagnostic curves tracking training loss (with standard deviation fills),
        validation accuracy, and processing execution time per epoch.
    Purpose:
        To support post-training checks, verify convergence rates, and diagnose potential underfitting/overfitting.
    Input Types:
        - epochs (list or np.ndarray): Integer epoch counts/indexes.
        - train_loss_means (list or np.ndarray): Numeric average training loss value per epoch.
        - train_loss_stds (list or np.ndarray): Standard deviation values of training loss per epoch.
        - val_accuracies (list or np.ndarray): Floating-point validation accuracy scores per epoch.
        - epoch_times (list or np.ndarray): Execution latency in seconds per epoch.
        - plot_folder (str): Target directory to save generated line plots.
    Output Types:
        - None
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
    Description:
        Loads the best saved checkpoint of the GNN model, computes inference over the test dataset,
        reports final accuracy, plots a normalized confusion matrix, and computes average batch inference latency.
    Purpose:
        To evaluate model performance, class-wise generalization capability, and production execution efficiency.
    Input Types:
        - gnn_model (nn.Module): The GNN model architecture.
        - test_loader (DataLoader): PyTorch Geometric DataLoader for testing data.
        - model_save_path (str): Filepath location to the best saved model state dictionary (.pt).
        - device (torch.device): CUDA or CPU execution hardware context.
        - label_encoder (LabelEncoder): Scikit-learn fitted label encoder for decoding integer class predictions back to actual category strings.
        - plot_folder (str): Target directory where the confusion matrix visualization is saved.
    Output Types:
        - accuracy (float): Computed test classification accuracy.
        - avg_infer_time (float): Average inference duration in seconds per data batch.
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
    Description:
        Constructs a temporal graph from a single-channel raw signal by extracting sliding windows as node features
        and linking all extracted nodes bidirectionally to represent a fully connected graph topology.
    Purpose:
        To transform a raw unimodal time-series waveform into a PyTorch Geometric Graph structure suitable for GNN layers.
    Input Types:
        - d1_sample (np.ndarray): 1D array representing the raw time-series sequence.
        - label (int): Integer encoded composition class label.
        - window_size (int): Temporal duration length of each sliding window (default: 500).
        - stride (int): Step stride spacing between consecutive sliding windows (default: 250).
    Output Types:
        - graph (torch_geometric.data.Data): PyG graph object containing node features (x), edge lists (edge_index), and class target (y).
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
    """
    Description:
        Computes and prints descriptive statistics about a list of graphs, including node count range and average nodes per graph.
    Purpose:
        To inspect graph sizes and verify that structural preprocessing was correctly applied.
    Input Types:
        - graphs (list of torch_geometric.data.Data): Pre-constructed Graph data list.
        - name (str): Dataset description label (e.g., 'Train' or 'Test').
    Output Types:
        - None
    """
    num_nodes = [g.num_nodes for g in graphs]
    print(f"📊 {name} Graph Stats")
    print(f"  - Total graphs: {len(graphs)}")
    print(f"  - Nodes per graph: min={min(num_nodes)
                                      }, max={max(num_nodes)}, avg={np.mean(num_nodes):.2f}")
