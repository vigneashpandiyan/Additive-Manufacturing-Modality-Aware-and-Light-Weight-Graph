# -*- coding: utf-8 -*-
"""Utility functions for data processing, visualization aids, and model setup.

This module contains helper functions for:
- Setting random seeds for reproducibility on GPU.
- Drawing an illustrative temporal graph representation (for docs/figures).
- Basic data transforms (normalize / standardize).
- Converting dual-channel time series into a PyG graph via sliding windows.
- Quick graph stats printout.

Notes
-----
- Public function behavior is preserved. Minor guards were added to avoid
  divide-by-zero in normalization/standardization.
- `create_shapelet_graph_batched` builds a fully-connected, bidirectional graph
  over sliding windows; node features are [2, window_size] (AE & Back-reflection).
  
Any reuse of this code should be authorized by the code author.
Developed for the publication:
"Modality-Aware and Light-Weight Graph Attention Networkfor In-SituComposition Monitoring 
in PBF-LB of Graded 316L–CuCrZr Alloys by Sensor Fusion of Optical and Acoustic Emissions"

"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.utils import resample  # noqa: F401 (kept for downstream use)
from sklearn.preprocessing import LabelEncoder  # noqa: F401
from sklearn.model_selection import train_test_split  # noqa: F401
from torch_geometric.data import Data
from Config import plot_folder


# -----------------------------------------------------------------------------
# Visual guide: Temporal graph diagram
# -----------------------------------------------------------------------------

def plot_temporal_graph_representation() -> None:
    """
    Visualize an illustrative temporal graph with:
    - 5 primary nodes (1–5) + 3 continuation nodes (6–8)
    - Triangles inside nodes for channels: Acoustic (▲) / Back Reflection (▼)
    - Forward (green) and backward (violet) directed edges between primary nodes
    - Continuation edges forward into "empty" nodes
    - Labels above primary nodes with time spans; faint labels for continuation

    This is meant for documentation/figures to convey the temporal connectivity
    you use elsewhere (it does not feed the model).
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


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_gpu_seed(seed: int = 42) -> None:
    """Set random seed for CUDA operations, if available.

    Parameters
    ----------
    seed : int
        Seed to use across CUDA devices.

    Notes
    -----
    - Silently no-ops if CUDA is not available.
    """
    try:
        torch.cuda.manual_seed_all(seed)
        # Collect interprocess handles; harmless if not needed.
        torch.cuda.ipc_collect()
    except Exception:
        # CUDA not available or context not initialized
        pass


# -----------------------------------------------------------------------------
# Data transforms
# -----------------------------------------------------------------------------

def normalize(data: np.ndarray) -> np.ndarray:
    """Min–Max normalize an array to [-1, 1].

    Parameters
    ----------
    data : np.ndarray
        Input array of any shape.

    Returns
    -------
    np.ndarray
        Normalized array in [-1, 1].

    Notes
    -----
    - Adds a small epsilon if max == min to avoid division by zero.
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
    """Standardize an array to zero mean and unit variance.

    Parameters
    ----------
    data : np.ndarray
        Input array of any shape.

    Returns
    -------
    np.ndarray
        Standardized array with ~zero mean and ~unit variance.

    Notes
    -----
    - Adds a small epsilon if std == 0 to avoid division by zero.
    """
    print("[STANDARDIZATION] Performing standardization")
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        std = 1e-12
    standardized = (data - mean) / std
    return standardized


# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------

def create_shapelet_graph_batched(
    d1_sample: np.ndarray,
    d2_sample: np.ndarray,
    label: int,
    window_size: int = 500,
    stride: int = 250,
) -> Data:
    """
    Construct a fully-connected bidirectional graph from two-channel time series
    via sliding windows.

    Each window of length `window_size` becomes a node whose feature is a tensor
    of shape [2, window_size] (channel-first: AE index 0, Back-reflection index 1).
    All node pairs are connected in both directions (i != j).

    Parameters
    ----------
    d1_sample : np.ndarray, shape (T,)
        Channel-1 time series (Acoustic Emission).
    d2_sample : np.ndarray, shape (T,)
        Channel-2 time series (Back-reflection).
    label : int
        Graph-level class label.
    window_size : int
        Sliding window length.
    stride : int
        Step size between window starts.

    Returns
    -------
    Data
        PyG Data with:
            x : FloatTensor [num_nodes, 2, window_size]
            edge_index : LongTensor [2, num_edges]
            y : LongTensor []  (scalar class label)

    Notes
    -----
    - No self-loops are added (i != j). If desired, you can add them later.
    - Windows are contiguous segments; no padding is applied at the tail.
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


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

def graph_stats(graphs: Sequence[Data], name: str = "") -> None:
    """Print a quick summary of node counts across a list of graphs.

    Parameters
    ----------
    graphs : Sequence[Data]
        Iterable of PyG graphs.
    name : str
        Optional label printed in the header.
    """
    num_nodes = [int(g.num_nodes) for g in graphs]
    if not num_nodes:
        print(f"📊 {name} Graph Stats\n  - No graphs provided.")
        return
    print(f"📊 {name} Graph Stats")
    print(f"  - Total graphs: {len(graphs)}")
    print(f"  - Nodes per graph: min={min(num_nodes)
                                      }, max={max(num_nodes)}, avg={np.mean(num_nodes):.2f}")
