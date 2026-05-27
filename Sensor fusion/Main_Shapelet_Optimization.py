# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Optimizing shapelet parameters (number of shapelets, shapelet lengths) iteratively.
- Tracking parameter sizes and classification accuracies to identify the most lightweight config.

Note: Any reuse of this code should be authorized by the code author.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import DataLoader

# Add parent and current directory to sys.path for direct relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Config import SEED, data_folder, plot_folder, num_epochs, batch_size, test_size, val_size, shapelet_len
# Override plot_folder specifically for shapelet optimization outputs
plot_folder = os.path.join(parent_dir, 'Sensor fusion', 'Figures', 'Shaplet_Optimization')

from network import GNNWithAttention
from utils import create_shapelet_graph_batched, standardize, set_gpu_seed, graph_stats
from dataloader import load_and_preprocess_data, create_graph_dataset

# Check for thop library (for profiling DL models)
try:
    import thop
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


def estimate_flops_analytical(num_shapelets, shapelet_len=50, num_nodes=19, window_size=500):
    """
    Description:
        Analytically estimates total model FLOPs count based on shapelet matches and GAT layer dimensions.
    Purpose:
        To calculate FLOP count mathematically for complexity sweeps.
    Input Types:
        - num_shapelets (int): Number of shapelets per channel.
        - shapelet_len (int): Length of each shapelet. Default is 50.
        - num_nodes (int): Number of nodes in the graph. Default is 19.
        - window_size (int): Temporal window width. Default is 500.
    Output Types:
        - estimated_flops (int): Sum analytical FLOPs count.
    """
    L = shapelet_len
    T = window_size
    shapelet_extraction_flops = 2 * num_nodes * num_shapelets * (T - L + 1) * (3 * L - 1)
    gat_flops = num_nodes * (num_shapelets * 2 * 16 * 2 + 32 * 32 + 32 * 16)
    return int(shapelet_extraction_flops + gat_flops)


def get_flops_profile(num_shapelets, num_classes, device):
    """
    Description:
        Profiles the GNN model using the `thop` library combined with analytical shapelet match calculation.
    Purpose:
        To provide complete FLOP profiles representing both the shapelet matching layer and structural attention layers.
    Input Types:
        - num_shapelets (int): Number of shapelets per channel.
        - num_classes (int): Total output composition classes.
        - device (torch.device): Computation device.
    Output Types:
        - total_flops (int): Total FLOP complexity.
    """
    num_nodes = 19
    # Compute the analytical shapelet matching FLOPs
    L = shapelet_len
    T = 500
    shapelet_flops = 2 * num_nodes * num_shapelets * (T - L + 1) * (3 * L - 1)

    if THOP_AVAILABLE:
        try:
            # Instantiate a dummy model to avoid modifying the trained model in-place
            dummy_model = GNNWithAttention(
                in_channels=num_shapelets * 2,
                hidden_channels=16,
                out_channels=num_classes,
                shapelet_len=shapelet_len,
                num_shapelets=num_shapelets
            ).to(device)
            dummy_model.eval()
            
            x = torch.randn(num_nodes, 2, 500).to(device)
            # Standard bidirectional edges
            src, dst = [], []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        src.append(i)
                        dst.append(j)
            edge_index = torch.tensor([src, dst], dtype=torch.long).to(device)
            batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
            
            flops, _ = thop.profile(dummy_model, inputs=(x, edge_index, batch), verbose=False)
            del dummy_model
            # Return thop profiled GAT + MLP layers + analytical shapelet operations
            return int(flops) + int(shapelet_flops)
        except Exception as e:
            print(f"  [FLOPs] thop profiling error: {e}. Falling back to pure analytical formula.")
            return estimate_flops_analytical(num_shapelets)
    else:
        return estimate_flops_analytical(num_shapelets)


def train_and_evaluate_model(num_shapelets, num_classes, train_loader, val_loader, test_loader, device):
    """
    Description:
        Initializes a GNN model with the specified shapelet count, trains it across all epochs, records validation loss trends to save the best checkpoint, and measures test accuracy, Macro F1, training speed, and sample inference latency.
    Purpose:
        To serve as the core evaluation unit within the parameter optimization sweep.
    Input Types:
        - num_shapelets (int): Number of shapelets to evaluate.
        - num_classes (int): Number of alloy classes.
        - train_loader (DataLoader): Training loader.
        - val_loader (DataLoader): Validation loader.
        - test_loader (DataLoader): Test loader.
        - device (torch.device): target computational device.
    Output Types:
        - metrics (dict): Dictionary of accuracy, f1_score, parameters, flops, epoch time, and inference latency.
    """
    # Reset random seeds to ensure identical weight initialization and shuffling order as Main.py
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_gpu_seed(SEED)
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize GNN Model
    model = GNNWithAttention(
        in_channels=num_shapelets * 2,
        hidden_channels=16,
        out_channels=num_classes,
        shapelet_len=shapelet_len,
        num_shapelets=num_shapelets
    ).to(device)

    # Count parameters and profile FLOPs
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops_count = get_flops_profile(num_shapelets, num_classes, device)

    # Optimization config - same learning rate and criterion as Main.py
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_model_path = f"best_model_sweep_k{num_shapelets}.pt"
    
    # Standard training loop
    start_train_time = time.time()
    epoch_times = []
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_times.append(time.time() - epoch_start)

        # Validation Step
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                preds = out.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        avg_loss = total_loss / len(train_loader)
        # Advance the PyTorch random state identically by running the standard deviation computation (with active dropout forward passes)
        _ = np.std([loss.item() for batch in train_loader for loss in [criterion(
            model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device)),
            batch.y.to(device))]])

        if epoch == 1 or epoch % 10 == 0 or epoch == num_epochs:
            print(f"      Epoch {epoch:03d}/{num_epochs:03d} | Avg Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Epoch Time: {epoch_times[-1]:.2f}s")

    total_training_time = time.time() - start_train_time
    avg_epoch_time = np.mean(epoch_times)

    # Load best model checkpoint for evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Test Set Evaluation
    test_preds, test_labels, infer_latencies = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Precise batch latency measurement
            t0 = time.time()
            out = model(batch.x, batch.edge_index, batch.batch)
            latency = time.time() - t0
            
            preds = out.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch.y.cpu().numpy())
            
            # Latency per sample in ms
            infer_latencies.append((latency / batch.y.size(0)) * 1000.0)

    # Calculate metrics
    acc = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    avg_infer_latency = np.mean(infer_latencies)

    # Clean up checkpoint file
    if os.path.exists(best_model_path):
        os.remove(best_model_path)

    del model; gc.collect(); torch.cuda.empty_cache()

    return {
        "accuracy": acc * 100.0,
        "f1_score": f1 * 100.0,
        "parameters": params_count,
        "flops": flops_count,
        "train_time_epoch": avg_epoch_time,
        "infer_latency": avg_infer_latency
    }


def main():
    """
    Description:
        Sets up the environment, loads datasets, runs the systematic sweep across shapelet counts `[2, 4, 6, 8, 10, 12, 14, 16]`, prints progress logs, and saves publication-ready visual evaluation reports to disk.
    Purpose:
        To execute the overall optimization sweep.
    Input Types:
        - None
    Output Types:
        - None
    """
    # Initialize reproducibility (exactly as in Main.py)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_gpu_seed(SEED)
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("SHAPELET NUMBER OPTIMIZATION SWEEP AND COMPLEXITY ANALYZER ")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Training Epochs: {num_epochs} | LR: 0.005 | Batch Size: {batch_size}")
    print("=" * 70)

    # Load data
    D1, D2, Y, class_labels, num_classes, label_encoder = load_and_preprocess_data()

    # Convert to graphs
    graph_list = create_graph_dataset(D1, D2, Y)

    # Train, validation, test splits (65% Train, 15% Val, 20% Test) - identical to Main.py
    train_val_graphs, test_graphs = train_test_split(
        graph_list, test_size=test_size, stratify=Y, random_state=42)
    train_val_labels = [g.y.item() for g in train_val_graphs]
    val_ratio = val_size / (1.0 - test_size)
    train_graphs, val_graphs = train_test_split(
        train_val_graphs, test_size=val_ratio, stratify=train_val_labels, random_state=42)

    print(f"[SPLIT] Partitioned graphs (identical to Main.py): Train={len(train_graphs)} | Val={len(val_graphs)} | Test={len(test_graphs)}")
    graph_stats(train_graphs, "Train Dataset")

    # Data loaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # Sweep configuration
    shapelet_sweep_range = [2, 4, 6, 8, 10, 12, 14, 16]
    results = []

    print("\n" + "#" * 60)
    print(" RUNNING SHAPELET COUNT OPTIMIZATION SWEEP ")
    print("#" * 60)

    for k in shapelet_sweep_range:
        print(f"\n [SWEEP] Evaluating GNN model with {k} shapelets per channel...")
        run_res = train_and_evaluate_model(
            num_shapelets=k,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device
        )
        print(f"    Accuracy: {run_res['accuracy']:.2f}% | F1: {run_res['f1_score']:.2f}%")
        print(f"    Parameters: {run_res['parameters']:,} | FLOPs: {run_res['flops']:,}")
        print(f"    Train: {run_res['train_time_epoch']:.3f} s/epoch | Latency: {run_res['infer_latency']:.4f} ms/sample")
        
        results.append({
            "num_shapelets": k,
            **run_res
        })

    # Convert to DataFrame
    df_res = pd.DataFrame(results)

    # Save results as CSV
    os.makedirs(plot_folder, exist_ok=True)
    csv_path = os.path.join(plot_folder, "shapelet_optimization_results.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\n Saved sweep metrics database to {csv_path}")

    # Generate ASCII report
    print("\n" + "=" * 105)
    print("FINAL COMPARATIVE COMPLEXITY AND ACCURACY SWEEP REPORT 🏆")
    print("=" * 105)
    print(f"{'Shapelets/Ch':<14} | {'Accuracy (%)':<14} | {'F1-Score (%)':<14} | {'Parameters':<12} | {'FLOPs Count':<14} | {'Epoch Time':<12} | {'Latency':<12}")
    print("-" * 105)
    for idx, row in df_res.iterrows():
        print(f"{int(row['num_shapelets']):<14} | "
              f"{row['accuracy']:>12.2f}% | "
              f"{row['f1_score']:>12.2f}% | "
              f"{int(row['parameters']):>12,} | "
              f"{int(row['flops']):>14,} | "
              f"{row['train_time_epoch']:>11.3f}s | "
              f"{row['infer_latency']:>10.3f}ms")
    print("=" * 105 + "\n")

    # =============================================================================
    # Generate Publication-Ready Individual Plots
    # =============================================================================
    print("[VISUALIZATION] Rendering publication-ready figures...")
    
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    
    ks = df_res["num_shapelets"]
    os.makedirs(plot_folder, exist_ok=True)

    # Colors
    c_blue = "#1F77B4"
    c_green = "#2CA02C"
    c_orange = "#FF7F0E"
    c_red = "#D62728"
    c_purple = "#9467BD"

    # --- Plot 1: Accuracy & F1-Score vs. Shapelets ---
    fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
    lns1 = ax1.plot(ks, df_res["accuracy"], marker="o", color=c_blue, linewidth=2, label="Accuracy")
    ax1.set_xlabel("Number of Shapelets (per modality)", fontsize=12)
    ax1.set_ylabel("Classification Accuracy (%)", fontsize=12, color=c_blue)
    ax1.tick_params(axis="y", labelcolor=c_blue, labelsize=10)
    ax1.set_title("Accuracy & F1-Score vs. Shapelets", fontsize=13)
    ax1.set_xticks(shapelet_sweep_range)

    ax1_twin = ax1.twinx()
    lns2 = ax1_twin.plot(ks, df_res["f1_score"], marker="s", color=c_green, linewidth=2, label="Macro F1-Score")
    ax1_twin.set_ylabel("Macro F1-Score (%)", fontsize=12, color=c_green)
    ax1_twin.tick_params(axis="y", labelcolor=c_green, labelsize=10)

    # Consolidated legend
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="lower right", framealpha=0.9, fontsize=10)
    fig1.tight_layout()
    plot_path1 = os.path.join(plot_folder, "shapelet_accuracy_f1.png")
    fig1.savefig(plot_path1, bbox_inches="tight", dpi=350, facecolor="white")
    plt.close(fig1)

    # --- Plot 2: Trainable Parameters & FLOPs vs. Shapelets ---
    fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=300)
    lns3 = ax2.plot(ks, df_res["parameters"], marker="p", color=c_orange, linewidth=2, label="Parameters")
    ax2.set_xlabel("Number of Shapelets (per modality)", fontsize=12)
    ax2.set_ylabel("Trainable Parameters Count", fontsize=12, color=c_orange)
    ax2.tick_params(axis="y", labelcolor=c_orange, labelsize=10)
    ax2.set_title("Parameter Size & FLOPs complexity", fontsize=13)
    ax2.set_xticks(shapelet_sweep_range)

    ax2_twin = ax2.twinx()
    lns4 = ax2_twin.plot(ks, df_res["flops"], marker="^", color=c_red, linewidth=2, label="FLOPs")
    ax2_twin.set_ylabel("Model FLOPs count", fontsize=12, color=c_red)
    ax2_twin.tick_params(axis="y", labelcolor=c_red, labelsize=10)

    lns_c = lns3 + lns4
    labs_c = [l.get_label() for l in lns_c]
    ax2.legend(lns_c, labs_c, loc="upper left", framealpha=0.9, fontsize=10)
    fig2.tight_layout()
    plot_path2 = os.path.join(plot_folder, "shapelet_parameters_flops.png")
    fig2.savefig(plot_path2, bbox_inches="tight", dpi=350, facecolor="white")
    plt.close(fig2)

    # --- Plot 3: Inference Latency vs. Shapelets ---
    fig3, ax3 = plt.subplots(figsize=(6, 4), dpi=300)
    ax3.plot(ks, df_res["infer_latency"], marker="h", color=c_purple, linewidth=2.5)
    ax3.set_xlabel("Number of Shapelets (per modality)", fontsize=12)
    ax3.set_ylabel("Inference Latency per Sample (ms)", fontsize=12)
    ax3.set_title("Inference Latency vs. Shapelets", fontsize=13)
    ax3.set_xticks(shapelet_sweep_range)
    ax3.tick_params(labelsize=10)
    fig3.tight_layout()
    plot_path3 = os.path.join(plot_folder, "shapelet_latency.png")
    fig3.savefig(plot_path3, bbox_inches="tight", dpi=350, facecolor="white")
    plt.close(fig3)

    # --- Plot 4: Pareto Front (Accuracy vs. FLOPs Complexity) ---
    fig4, ax4 = plt.subplots(figsize=(6, 4), dpi=300)
    flops_in_millions = df_res["flops"] / 1e6
    ax4.plot(flops_in_millions, df_res["accuracy"], color="gray", linestyle="-.", alpha=0.5)
    scatter = ax4.scatter(flops_in_millions, df_res["accuracy"], c=ks, cmap="bwr", s=150, zorder=5, edgecolor="black", linewidth=1.2)
    ax4.set_xlabel("Computational Complexity (Million FLOPs)", fontsize=12)
    ax4.set_ylabel("Classification Accuracy (%)", fontsize=12)
    ax4.set_title("Accuracy vs. Computational Complexity", fontsize=13)
    ax4.tick_params(labelsize=10)
    ax4.margins(x=0.15, y=0.15)  # Add margins to prevent markers/labels from going out of frame

    # Annotate points with the shapelet number to show the "elbow" clearly
    for idx, row in df_res.iterrows():
        ax4.annotate(
            f"k={int(row['num_shapelets'])}",
            xy=(row["flops"] / 1e6, row["accuracy"]),
            xytext=(7, -4),
            textcoords="offset points",
            fontsize=10
        )
        
    cbar = fig4.colorbar(scatter, ax=ax4)
    cbar.set_label("Shapelets Number (k)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig4.tight_layout()
    plot_path4 = os.path.join(plot_folder, "shapelet_pareto.png")
    fig4.savefig(plot_path4, bbox_inches="tight", dpi=350, facecolor="white")
    plt.close(fig4)

    print(f" Saved individual publication-ready visualization reports under {plot_folder}")
    plt.show()

    print("\n" + "#" * 60)
    print("SHAPELET SWEEP COMPLETED SUCCESSFULLY! ")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
