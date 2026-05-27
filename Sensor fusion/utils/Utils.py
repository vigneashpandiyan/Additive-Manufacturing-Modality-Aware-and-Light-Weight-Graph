# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Data normalization, standardization, graph construction from sliding sub-windows, and reproducible GPU random seeding.

Note: Any reuse of this code should be authorized by the code author.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import os
import sys

# Resolve parent directory to allow imports of Config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Config import plot_folder
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data


def plot_temporal_graph_representation() -> None:
    """
    Description:
        Constructs and visualizes an illustrative temporal graph diagram including node details, color-coded forward/backward directed edges, internal modality symbols (▲/▼), and temporal labels. Saves the diagram to disk.
    Purpose:
        To save illustrative explanation figures of the temporal connectivity layout used in the framework.
    Input Types:
        - None
    Output Types:
        - None: Directly draws and saves the diagram.
    """
    num_primary_nodes = 5
    num_empty_nodes = 3
    total_nodes = num_primary_nodes + num_empty_nodes

    # Directed edges among primary nodes (bi-directional)
    forward_edges = [(i, i + 1) for i in range(1, num_primary_nodes)]
    backward_edges = [(i + 1, i) for i in range(1, num_primary_nodes)]

    # Forward continuation from last primary to continuation nodes
    continuation_edges = [(i, i + 1)
                          for i in range(num_primary_nodes, total_nodes)]

    all_edges = forward_edges + backward_edges + continuation_edges

    # Build graph
    G = nx.DiGraph()
    for i in range(1, total_nodes + 1):  # Nodes numbered 1..8
        G.add_node(i)
    G.add_edges_from(all_edges)

    # Labels for primary nodes only
    node_text_labels = {
        1: "Node 1\nt=0 to t=Δt",
        2: "Node 2\nt=Δt to t=2Δt",
        3: "Node 3\nt=2Δt to t=3Δt",
        4: "Node 4\nt=3Δt to t=4Δt",
        5: "Node 5\nt=4Δt to t=5Δt",
    }

    # Colors: 5 primaries + 3 continuation
    node_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1',
                   '#FFA07A', '#98D8C8'] + ['white'] * num_empty_nodes

    # Edge colors by direction
    edge_colors = []
    for u, v in all_edges:
        # green fwd, violet bwd
        edge_colors.append('#4CAF50' if u < v else '#9C27B0')

    # Layout: diagonal progression
    plt.figure(figsize=(12, 8), dpi=100)
    pos = {}
    x_spacing = 1.0 / (total_nodes - 2)
    y_spacing = 1.0 / (total_nodes - 2)
    for i in range(1, total_nodes + 1):
        pos[i] = ((i - 1) * x_spacing, (i - 1) * y_spacing)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        edgecolors='black',
        linewidths=2,
        node_size=3500,
        alpha=0.9,
    )

    # Style continuation nodes (semi-transparent)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(range(num_primary_nodes + 1, total_nodes + 1)),
        node_color='gray',
        edgecolors='gray',
        linewidths=2,
        node_size=1500,
        alpha=0.4,
    )

    # Draw edges with curvature
    for (u, v), color in zip(all_edges, edge_colors):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            edge_color=color,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=25,
            width=2.5,
            connectionstyle='arc3,rad=0.25',
            node_size=3500,
            alpha=0.8,
        )

    # Channel indicators: triangles inside primary nodes
    for i in range(1, num_primary_nodes + 1):
        x, y = pos[i]
        # Acoustic Emission (▲)
        plt.text(x, y + 0.03, '▲', ha='center', va='center',
                 fontsize=14, color='#FF5722', fontweight='light')
        # Back Reflection (▼)
        plt.text(x, y - 0.03, '▼', ha='center', va='center',
                 fontsize=14, color='#3F51B5', fontweight='light')

    # Labels above primary nodes
    label_offset = 0.15
    for i in range(1, num_primary_nodes + 1):
        x, y = pos[i]
        plt.text(
            x, y + label_offset,
            node_text_labels[i],
            ha='center',
            va='bottom',
            fontsize=16,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='black',
                      boxstyle='round,pad=0.4', alpha=0.9, linewidth=1.2),
        )

    # Faint labels for continuation nodes
    for i in range(num_primary_nodes + 1, total_nodes + 1):
        x, y = pos[i]
        plt.text(
            x, y + label_offset,
            f"Node {i}\nt={i-1}Δt to t={i}Δt",
            ha='center',
            va='bottom',
            fontsize=9,
            color='gray',
            alpha=0.7,
            bbox=dict(facecolor='white', edgecolor='gray',
                      boxstyle='round,pad=0.3', alpha=0.6, linewidth=1),
        )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#4CAF50', lw=3, label='Forward (1→n)'),
        Line2D([0], [0], color='#9C27B0', lw=3, label='Backward (n→1)'),
        Line2D([0], [0], marker='^', color='w', label='Acoustic emission (Ch1)',
               markerfacecolor='#FF5722', markersize=18),
        Line2D([0], [0], marker='v', color='w', label='Back reflection (Ch2)',
               markerfacecolor='#3F51B5', markersize=18),
    ]
    legend = plt.legend(
        handles=legend_elements,
        loc='upper left',
        framealpha=0.9,
        prop={'size': 18, 'weight': 'light'},
        handlelength=2,
        labelspacing=1,
    )
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_edgecolor('black')

    # Title + timeline
    plt.title("Temporal graph with bi-directional connections",
              fontsize=25, pad=25, fontweight='light')
    plt.annotate(
        'Temporal Progression →',
        xy=(0.5, 0.4), xycoords='axes fraction',
        ha='center', va='center',
        fontsize=20, rotation=30, rotation_mode='anchor',
    )

    # Styling
    plt.xlim(-0.1, 1.25)
    plt.ylim(-0.1, 1.25)
    plt.grid(True, alpha=0.03)
    plt.axis('off')
    plt.tight_layout()

    # Save
    out_path = os.path.join(
        plot_folder, "temporal_graph_representation_enhanced.jpg")
    plt.savefig(out_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.show()
    plt.close()


def set_gpu_seed(seed: int = 42) -> None:
    """
    Description:
        Sets random seeds for CUDA devices and runs CUDA interprocess collections.
    Purpose:
        To establish reproducibility on GPU accelerators.
    Input Types:
        - seed (int): Seed number. Default is 42.
    Output Types:
        - None
    """
    try:
        torch.cuda.manual_seed_all(seed)
        # Collect interprocess handles; harmless if not needed.
        torch.cuda.ipc_collect()
    except Exception:
        # CUDA not available or context not initialized
        pass


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Description:
        Min–Max normalizes the values of an array to the interval [-1, 1], avoiding divide-by-zero errors.
    Purpose:
        To normalize raw signal values.
    Input Types:
        - data (numpy.ndarray): Multi-dimensional array.
    Output Types:
        - normalized (numpy.ndarray): Normalized array.
    """
    print("[NORMALIZATION] Performing Min-Max normalization to [-1, 1]")
    data_min = np.min(data)
    data_max = np.max(data)
    denom = (data_max - data_min)
    if denom == 0:
        denom = 1e-12
    normalized = 2 * ((data - data_min) / denom) - 1
    return normalized


def standardize(data: np.ndarray) -> np.ndarray:
    """
    Description:
        Standardizes the values of an input array to zero mean and unit variance.
    Purpose:
        To stabilize input scaling across sensory modalities.
    Input Types:
        - data (numpy.ndarray): Input array.
    Output Types:
        - standardized (numpy.ndarray): Standardized array.
    """
    print("[STANDARDIZATION] Performing standardization")
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        std = 1e-12
    standardized = (data - mean) / std
    return standardized


def create_shapelet_graph_batched(
    d1_sample: np.ndarray,
    d2_sample: np.ndarray,
    label: int,
    window_size: int = 500,
    stride: int = 250,
) -> Data:
    """
    Description:
        Extracts sliding temporal windows from two input signals, constructs windowed node feature matrices, maps fully connected bidirectional edge index matrices, and wraps them inside PyTorch Geometric `Data` graphs.
    Purpose:
        To form graph representations for GNN spatial-temporal processing.
    Input Types:
        - d1_sample (numpy.ndarray): Acoustic emission signal array.
        - d2_sample (numpy.ndarray): Optical emission signal array.
        - label (int): Alloy composition target index.
        - window_size (int): Segment window length. Default is 500.
        - stride (int): Step size. Default is 250.
    Output Types:
        - Graph (torch_geometric.data.Data): Graph containing window node tensors and edge index maps.
    """
    # Basic input checks
    assert d1_sample.ndim == 1 and d2_sample.ndim == 1, "Inputs must be 1D arrays."
    assert d1_sample.shape[0] == d2_sample.shape[0], "Channel lengths must match."
    assert window_size > 0 and stride > 0, "window_size and stride must be positive."

    T = d1_sample.shape[0]
    windows: List[np.ndarray] = []

    # Build windows
    for start in range(0, T - window_size + 1, stride):
        d1_window = d1_sample[start:start + window_size]
        d2_window = d2_sample[start:start + window_size]
        window = np.stack([d1_window, d2_window], axis=0)  # [2, window_size]
        windows.append(window)

    # Convert to tensor: [num_nodes, 2, window_size]
    if len(windows) == 0:
        # Edge case: if series is shorter than window_size
        raise ValueError(
            "No windows were created. Decrease window_size or stride.")

    window_tensor = torch.tensor(np.stack(windows, axis=0), dtype=torch.float)
    num_nodes = window_tensor.shape[0]

    # Create fully connected, bidirectional edges (i != j)
    # This is O(n^2); for many nodes consider sparse constructions.
    edge_src = []
    edge_dst = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_src.append(i)
                edge_dst.append(j)
    edge_index = torch.tensor([edge_src, edge_dst],
                              dtype=torch.long, device=window_tensor.device)

    y = torch.tensor(label, dtype=torch.long)

    return Data(x=window_tensor, edge_index=edge_index, y=y)


def graph_stats(graphs: Sequence[Data], name: str = "") -> None:
    """
    Description:
        Calculates and prints out statistics (minimum, maximum, mean) of node counts across a sequence of PyG Graph samples.
    Purpose:
        To log dataset properties.
    Input Types:
        - graphs (Sequence[Data]): Array of Graph samples.
        - name (str): Display header name. Default is "".
    Output Types:
        - None
    """
    num_nodes = [int(g.num_nodes) for g in graphs]
    if not num_nodes:
        print(f"📊 {name} Graph Stats\n  - No graphs provided.")
        return
    print(f"📊 {name} Graph Stats")
    print(f"  - Total graphs: {len(graphs)}")
    print(f"  - Nodes per graph: min={min(num_nodes)
                                      }, max={max(num_nodes)}, avg={np.mean(num_nodes):.2f}")
