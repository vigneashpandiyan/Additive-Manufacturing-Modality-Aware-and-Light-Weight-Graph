# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Standard training loops, validation scoring, model checkpointing, testing evaluations, and curve plotting utilities.

Note: Any reuse of this code should be authorized by the code author.
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
    Description:
        Runs the full model training process: iterates over epochs, executes forward and backward passes, calculates loss standard deviations, saves checkpoints of the best model, and calls plotting utilities.
    Purpose:
        To optimize model weights dynamically during framework training.
    Input Types:
        - model (GNNWithAttention): Model to train.
        - train_loader (DataLoader): PyG DataLoader containing the training set.
        - test_loader (DataLoader): PyG DataLoader containing the test set (used for validation).
        - optimizer (torch.optim.Optimizer): Optimizer algorithm model.
        - criterion (torch.nn.modules.loss._Loss): Criterion loss model.
        - device (torch.device): computational device target.
        - class_labels (list): String class labels names array.
    Output Types:
        - tuple: (best_model, model_save_path)
            - best_model (GNNWithAttention): Best performing checkpoint.
            - model_save_path (str): File path where parameters are saved.
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
    Description:
        Counts the total number of trainable parameter weights within the given model.
    Purpose:
        To calculate and report model capacity metrics.
    Input Types:
        - model (torch.nn.Module): Target PyTorch network model.
    Output Types:
        - parameter_count (int): Sum trainable parameters count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model, loader, device):
    """
    Description:
        Evaluates the model on a target dataset, measuring classification accuracy under eval mode with gradient tracking disabled.
    Purpose:
        To test model accuracy on unseen subsets during optimization.
    Input Types:
        - model (torch.nn.Module): Trained model to evaluate.
        - loader (DataLoader): Loader containing evaluation data.
        - device (torch.device): target computational device.
    Output Types:
        - accuracy (float): Classification accuracy score.
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
    Description:
        Generates and exports three PNG plots depicting: (1) training loss with batch-level standard deviation shading, (2) validation accuracy over epochs, and (3) computation time per epoch.
    Purpose:
        To save visual learning logs for manuscript validation.
    Input Types:
        - train_loss_means (list): Average epoch training losses.
        - train_loss_stds (list): Batch standard deviations.
        - val_accuracies (list): Validation accuracies per epoch.
        - epoch_times (list): Training seconds per epoch.
    Output Types:
        - None: Saves PNG figures to Figures folder.
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
    Description:
        Evaluates a trained model on a test dataset, measures inference timing, prints a classification report, and exports absolute and normalized confusion matrices.
    Purpose:
        To perform comprehensive framework validation testing and save final reporting outputs.
    Input Types:
        - model (torch.nn.Module): Trained model to test.
        - test_loader (DataLoader): Loader containing test data.
        - device (torch.device): computational device target.
        - label_encoder (LabelEncoder): Fitted encoder model used to map labels back to category names.
    Output Types:
        - tuple: (all_preds, all_labels)
            - all_preds (torch.Tensor): Output model predictions.
            - all_labels (torch.Tensor): Ground truth targets.
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
