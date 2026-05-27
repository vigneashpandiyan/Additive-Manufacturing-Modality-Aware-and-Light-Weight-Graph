# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- A single-modality Graph Neural Network using only acoustic (D2) or optical (D1) shapelet characteristics to classify alloy compositions.
- Environment setup, data parsing, standard data normalizations, and GNN training/evaluation workflows for unimodal sensor baselines.

Note: Any reuse of this code should be authorized by the code author.
"""
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample


from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool


from tqdm import tqdm
import time
import gc
from matplotlib import cm

from utils import *
from trainer import *


Sensor_stream = ["D1", "D2"]
for Material in Sensor_stream:

    print(f"\n🔧 Processing Sensor Stream: {Material}")

    # === Configuration and Setup ===
    SEED = 20  # Random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gc.collect()
    torch.cuda.empty_cache()
    set_gpu_seed()

    # === Folder Setup ===
    data_folder = "data"  # Directory containing input data files
    num_epochs = 300     # Number of training epochs
    batch_size = 256      # Batch size for training
    test_size = 0.3       # Proportion of data to use for testing
    shapelet_len = 50    # Length of learned shapelets
    num_shapelets = 20   # Number of shapelets per channel
    base_dir = os.getcwd()

    plot_folder = os.path.join(
        base_dir, 'Figures_30percent_20shaplets_16Dim', Material)
    os.makedirs(plot_folder, exist_ok=True)

    # === Load Data ===
    data_path = os.path.join(data_folder, f"{Material}_rawspace_5000.npy")
    D1_full = np.load(data_path)
    Y_full = np.load(os.path.join(data_folder, "classspace_5000.npy"))

    # === Class Balancing & Sampling ===
    # Perform balanced sampling to ensure equal representation of all classes
    samples_per_class = int(1 * len(Y_full) / len(np.unique(Y_full)))
    selected_indices = []
    min_count = min(np.sum(Y_full == c) for c in np.unique(Y_full))

    for cls in np.unique(Y_full):
        idx = np.where(Y_full == cls)[0]
        n = min(samples_per_class, len(idx), min_count)
        selected = resample(idx, replace=True, n_samples=n, random_state=SEED)
        selected_indices.extend(selected)

    D1 = D1_full[selected_indices]
    Y = Y_full[selected_indices]

    # Encode class labels as integers
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    class_labels = label_encoder.classes_
    num_classes = len(class_labels)

    D1 = standardize(D1)

    # === Build Graph Dataset ===
    print("[BUILD] Constructing graph representations from time series...")
    graph_list = []

    for i in tqdm(range(len(D1))):
        graph = create_shapelet_graph_batched(D1[i], Y[i])
        graph_list.append(graph)

    # === Model Configuration ===
    # shapelet_model = BatchedShapeletExtractor(shapelet_len, num_shapelets)

    # Split into train and test sets
    train_graphs, test_graphs = train_test_split(
        graph_list, test_size=test_size, stratify=Y, random_state=42)

    graph_stats(train_graphs, "Train")
    graph_stats(test_graphs, "Test")

    train_loader = DataLoader(
        train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # === Model Initialization ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model = GNNWithAttention(
        in_channels=num_shapelets * 2,
        hidden_channels=16,  # Divides by 4
        out_channels=len(class_labels),
        shapelet_len=shapelet_len,
        num_shapelets=num_shapelets
    ).to(device)

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    # === Training Loop ===
    print(f"Starting training on {device}...")
    start_time = time.time()
    train_losses, val_accuracies, epoch_times = [], [], []
    train_loss_means, train_loss_stds = [], []

    model_save_path = os.path.join(plot_folder, 'best_model.pt')
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        gnn_model.train()
        total_loss = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            iter_start = time.time()

            # Forward pass
            optimizer.zero_grad()
            out = gnn_model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"    Batch {
                      batch_idx+1:03d} Iter Time: {time.time() - iter_start:.3f}s")

        # Calculate training statistics
        avg_loss = total_loss / len(train_loader)
        train_loss_means.append(avg_loss)
        train_loss_stds.append(np.std([loss.item() for batch in train_loader for loss in [criterion(gnn_model(
            batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device)), batch.y.to(device))]]))

        # Validation evaluation
        gnn_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for val_batch in test_loader:
                val_batch = val_batch.to(device)
                val_out = gnn_model(
                    val_batch.x, val_batch.edge_index, val_batch.batch)
                val_preds = val_out.argmax(dim=1)
                correct += (val_preds == val_batch.y).sum().item()
                total += val_batch.y.size(0)

        acc = correct / total
        train_losses.append(avg_loss)
        val_accuracies.append(acc)
        epoch_times.append(time.time() - epoch_start)

        print(f"Epoch {epoch:02d}, Loss: {avg_loss:.4f}, Val Acc: {
              acc:.4f}, Epoch Time: {epoch_times[-1]:.2f}s")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(gnn_model.state_dict(), model_save_path)

    print(f"Total Trainable Parameters: {count_parameters(gnn_model):,}")

    # === Training Visualization ===
    print("Generating training performance plots...")
    epochs = list(range(1, num_epochs + 1))

    plot_training_curves(
        epochs=epochs,
        train_loss_means=train_loss_means,
        train_loss_stds=train_loss_stds,
        val_accuracies=val_accuracies,
        epoch_times=epoch_times,
        plot_folder=plot_folder
    )

    # === Model Evaluation ===
    print("Evaluating on test set...")
    acc, infer_time = evaluate_gnn_model(
        gnn_model=gnn_model,
        test_loader=test_loader,
        model_save_path=model_save_path,
        device=torch.device('cuda'),
        label_encoder=label_encoder,
        plot_folder=plot_folder
    )
