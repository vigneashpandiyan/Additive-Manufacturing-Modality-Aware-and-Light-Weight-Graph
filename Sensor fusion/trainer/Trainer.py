# -*- coding: utf-8 -*-
"""Training and evaluation functions for the multimodal GAT model.

This module provides complete training and evaluation workflows for:
- Model training with progress tracking and checkpointing
- Performance evaluation on validation/test sets
- Metric visualization and reporting
- Model inference timing

Key Components:
1. Main training loop with batch processing
2. Model evaluation functions
3. Performance metric calculation and visualization
4. Model parameter counting
5. Comprehensive test reporting

Example Usage:
    >>> model = GNNWithAttention(...)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> criterion = nn.CrossEntropyLoss()
    >>> trained_model, model_path = train_model(model, train_loader, test_loader,
    ...                                        optimizer, criterion, device, class_labels)
    >>> test_preds, test_labels = test_model(trained_model, test_loader, 
    ...                                    device, label_encoder)
    
Any reuse of this code should be authorized by the code author.
Developed for the publication:
"Modality-Aware and Light-Weight Graph Attention Networkfor In-SituComposition Monitoring 
in PBF-LB of Graded 316L–CuCrZr Alloys by Sensor Fusion of Optical and Acoustic Emissions"
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from tqdm import tqdm
import gc
import os
import sys

# Resolve parent directory to allow imports of Config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Config import plot_folder, num_epochs, batch_size
from torch_geometric.loader import DataLoader


def train_model(model, train_loader, test_loader, optimizer, criterion, device, class_labels):
    """
    Train the GNN model with full training workflow.

    Args:
        model: The GNN model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for validation data
        optimizer: Optimization algorithm
        criterion: Loss function
        device: Computation device ('cuda' or 'cpu')
        class_labels: List of class names for reporting

    Returns:
        tuple: (best_model, model_save_path)
            best_model: Model with best validation performance
            model_save_path: Path to saved model checkpoint

    Training Process:
        1. Iterates through specified number of epochs
        2. Processes batches with forward/backward passes
        3. Tracks training loss and validation accuracy
        4. Saves best performing model
        5. Generates training metrics visualization
    """
    print(f"Starting training on {device}...")
    start_time = time.time()
    train_losses, val_accuracies, epoch_times = [], [], []
    train_loss_means, train_loss_stds = [], []

    model_save_path = os.path.join(plot_folder, 'best_model.pt')
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            iter_start = time.time()

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"    Batch {
                      batch_idx+1:03d} Iter Time: {time.time() - iter_start:.3f}s")

        avg_loss = total_loss / len(train_loader)
        train_loss_means.append(avg_loss)
        train_loss_stds.append(np.std([loss.item() for batch in train_loader for loss in [criterion(
            model(batch.x.to(device), batch.edge_index.to(
                device), batch.batch.to(device)),
            batch.y.to(device))]]))

        val_acc = evaluate_model(model, test_loader, device)
        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)
        epoch_times.append(time.time() - epoch_start)

        print(f"Epoch {epoch:02d}, Loss: {avg_loss:.4f}, Val Acc: {
              val_acc:.4f}, Epoch Time: {epoch_times[-1]:.2f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)

    print(f"Total Trainable Parameters: {count_parameters(model):,}")
    plot_training_metrics(train_loss_means, train_loss_stds,
                          val_accuracies, epoch_times)
    return model, model_save_path


def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.

    Args:
        model: PyTorch model to analyze

    Returns:
        int: Total number of trainable parameters

    Example:
        >>> model = GNNWithAttention(...)
        >>> print(f"Parameters: {count_parameters(model):,}")
        Parameters: 1,234,567
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model, loader, device):
    """
    Evaluate model performance on a given dataset.

    Args:
        model: Trained model to evaluate
        loader: DataLoader for evaluation data
        device: Computation device ('cuda' or 'cpu')

    Returns:
        float: Accuracy score on the evaluation data

    Note:
        Sets model to eval() mode during evaluation
        Uses torch.no_grad() to disable gradient computation
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total


def plot_training_metrics(train_loss_means, train_loss_stds, val_accuracies, epoch_times):
    """
    Generate and save training metric visualizations.

    Args:
        train_loss_means: List of mean training losses per epoch
        train_loss_stds: List of training loss standard deviations per epoch
        val_accuracies: List of validation accuracies per epoch
        epoch_times: List of epoch durations in seconds

    Saves:
        - Training loss curve with std deviation
        - Validation accuracy curve
        - Epoch timing curve

    Files are saved to the plot_folder directory as:
        - train_loss_curve.png
        - val_accuracy_curve.png
        - epoch_time_curve.png
    """
    epochs = list(range(1, len(train_loss_means) + 1))

    # Training loss plot
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
    plt.close()

    # Validation accuracy plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_accuracies, label='Val Accuracy', color='tab:green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'val_accuracy_curve.png'))
    plt.close()

    # Epoch timing plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, epoch_times, marker='o', color='tab:orange')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Time per Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'epoch_time_curve.png'))
    plt.close()


def test_model(model, test_loader, device, label_encoder):
    """
    Comprehensive model testing and reporting.

    Args:
        model: Trained model to test
        test_loader: DataLoader for test data
        device: Computation device ('cuda' or 'cpu')
        label_encoder: Fitted LabelEncoder for class labels

    Returns:
        tuple: (all_preds, all_labels)
            all_preds: Tensor of model predictions
            all_labels: Tensor of ground truth labels

    Outputs:
        - Prints test accuracy and classification report
        - Generates and saves confusion matrices
        - Reports inference timing
        - Shows normalized and absolute confusion matrices
    """
    model.eval()
    all_preds, all_labels = [], []
    infer_times = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            t0 = time.time()
            out = model(batch.x, batch.edge_index, batch.batch)
            infer_times.append(time.time() - t0)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(batch.y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    print("\n=== Test Results ===")
    print(f"Test Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

    # Generate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, "confusion_matrix.png"))
    plt.close()

    # Generate normalized confusion matrix
    cm_normalized = confusion_matrix(
        all_labels, all_preds, normalize='true') * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized,
                                  display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, values_format=".1f")
    plt.title("Confusion Matrix (normalized by row)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, "confusion_matrix_percent.png"))
    plt.close()

    # Print classification report
    target_names = [str(cls) for cls in label_encoder.classes_]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    print(f"\nAverage Inference Time per Batch: {np.mean(infer_times):.4f}s")

    return all_preds, all_labels
