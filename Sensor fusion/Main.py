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
# Libraries to import
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gc
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import DataLoader

from Config import SEED, data_folder, plot_folder, num_epochs, batch_size, test_size, val_size, shapelet_len, num_shapelets
from network import GNNWithAttention, BatchedShapeletExtractor
from trainer import train_model, test_model
from utils import *
from visualization import *
from dataloader import load_and_preprocess_data, create_graph_dataset

# Check for thop library (for profiling DL models)
try:
    import thop
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


def estimate_flops_analytical(num_shapelets, shapelet_len=50, num_nodes=19, window_size=500):
    L = shapelet_len
    T = window_size
    shapelet_extraction_flops = 2 * num_nodes * num_shapelets * (T - L + 1) * (3 * L - 1)
    gat_flops = num_nodes * (num_shapelets * 2 * 16 * 2 + 32 * 32 + 32 * 16)
    return int(shapelet_extraction_flops + gat_flops)


def get_flops_profile(num_shapelets, num_classes, device):
    num_nodes = 19
    # Compute the analytical shapelet matching FLOPs
    L = shapelet_len
    T = 500
    shapelet_flops = 2 * num_nodes * num_shapelets * (T - L + 1) * (3 * L - 1)

    if THOP_AVAILABLE:
        try:
            dummy_model = GNNWithAttention(
                in_channels=num_shapelets * 2,
                hidden_channels=16,
                out_channels=num_classes,
                shapelet_len=shapelet_len,
                num_shapelets=num_shapelets
            ).to(device)
            dummy_model.eval()
            
            x = torch.randn(num_nodes, 2, 500).to(device)
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
            return int(flops) + int(shapelet_flops)
        except Exception:
            return estimate_flops_analytical(num_shapelets)
    else:
        return estimate_flops_analytical(num_shapelets)


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
    # Train, validation, test splits (65% Train, 15% Val, 20% Test)
    train_val_graphs, test_graphs = train_test_split(
        graph_list, test_size=test_size, stratify=Y, random_state=42)
    train_val_labels = [g.y.item() for g in train_val_graphs]
    val_ratio = val_size / (1.0 - test_size)
    train_graphs, val_graphs = train_test_split(
        train_val_graphs, test_size=val_ratio, stratify=train_val_labels, random_state=42)

    graph_stats(train_graphs, "Train")
    graph_stats(val_graphs, "Validation")
    graph_stats(test_graphs, "Test")

    # Create data loaders
    train_loader = DataLoader(
        train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_graphs, batch_size=batch_size, shuffle=False)

    # Initialize and train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, optimizer, criterion = initialize_model(num_classes, device)
    model_trained, model_save_path = train_model(model, train_loader, val_loader, optimizer,
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

    # Evaluate model on Test set
    all_preds, all_labels = test_model(
        model_trained, test_loader, device, label_encoder)

    # Run visualizations
    run_visualizations(model_trained, test_graphs, all_preds,
                       all_labels, device, class_labels)
    print("All visualizations completed!")

    # --- Computational Complexity & Latency Profiling ---
    print("[PROFILING] Measuring complexity and latency...")
    model_trained.eval()
    t_start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _ = model_trained(batch.x, batch.edge_index, batch.batch)
    total_test_infer_time = time.time() - t_start
    avg_latency_per_sample_ms = (total_test_infer_time / len(test_graphs)) * 1000.0

    # Count trainable parameters and profile FLOPs
    params_count = sum(p.numel() for p in model_trained.parameters() if p.requires_grad)
    flops_count = get_flops_profile(num_shapelets, num_classes, device)

    # Print final complexity and resource metrics
    print("\n" + "=" * 80)
    print("📈 FINAL SYSTEM COMPUTATIONAL COMPLEXITY & RESOURCE ANALYSIS 📈")
    print("=" * 80)
    print(f"  - Data Splits:                Train = 65% | Val = 15% | Test = 20%")
    print(f"  - Total Trainable Parameters: {params_count:,} parameters")
    print(f"  - Model FLOPs Complexity:    {flops_count:,} FLOPs")
    print(f"  - Total Inference Time (Test): {total_test_infer_time:.4f} seconds")
    print(f"  - Average Inference Latency:  {avg_latency_per_sample_ms:.4f} ms per sample")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
