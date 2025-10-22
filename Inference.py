# -*- coding: utf-8 -*-
"""
Complete Inference and Visualization Script for Multimodal GAT with Shapelets.

This module provides utilities to (1) quantify class-wise channel contributions
(Acoustic vs Optical) using saliency backprop, and (2) compute/plot average
node-level saliency per class from a balanced subset of graphs.

Assumptions
-----------
- Each `Data` object holds:
    x: Tensor [num_nodes, 2, T]          # two channels: AE (0) and Optical (1)
    edge_index: LongTensor [2, num_edges]
    batch: LongTensor [num_nodes]        # node -> graph id
    y: LongTensor [num_graphs]           # graph-level class label
- The model forward signature is: model(x, edge_index, batch) -> logits [B, C]
  where B = number of graphs in the mini-batch, C = number of classes.

Notes
-----
- Saliency is computed via gradients of the class score w.r.t. inputs (x).
- Channel contributions are aggregated with global max pool over nodes.

Any reuse of this code should be authorized by the code author.
Developed for the publication:
"Modality-Aware and Light-Weight Graph Attention Networkfor In-SituComposition Monitoring 
in PBF-LB of Graded 316L–CuCrZr Alloys by Sensor Fusion of Optical and Acoustic Emissions"

"""

from collections import OrderedDict
from sklearn.utils import resample
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
from matplotlib import cm
from Config import plot_folder
import os
from torch_geometric.loader import DataLoader


def saliency_channel_contributions_per_class(model, correct_graphs, device, plot_folder, save_prefix):
    """
    Compute and visualize per-class channel contributions (AE vs Optical) using saliency.

    For each mini-batch of graphs, this function:
    1) Enables gradient on the inputs x and runs a forward pass.
    2) Backpropagates the score of the ground-truth class to obtain saliency.
    3) Sums saliency across time for each channel per node, then aggregates per graph
       with a global max pool across nodes (focuses on the most salient node).
    4) Averages graph-level contributions within each class, normalizes to percentages,
       and plots side-by-side bars with error bars.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model. Must accept (x, edge_index, batch) and return logits [B, C].
    correct_graphs : Sequence[torch_geometric.data.Data]
        Collection of correctly classified graphs (or graphs you want to analyze).
    device : torch.device or str
        Device on which to run the computations.
    plot_folder : str
        Directory to save the figure.
    save_prefix : str
        Filename prefix for the saved plot (PNG).

    Returns
    -------
    summary_dict : dict[int, dict[str, float]]
        Mapping class_id -> {'AE %', 'AE std', 'Optical %', 'Optical std'}.
        Percentages sum to ~100 within each class (AE% + Optical%).

    Notes
    -----
    - Uses global *max* pool over nodes to emphasize the most influential segment.
      If you prefer smoother aggregation, replace with global_mean_pool.
    - Uses population std (unbiased=False) for error bars.
    """
    model.eval()
    dataloader = DataLoader(correct_graphs, batch_size=256, shuffle=False)
    per_class_contrib = defaultdict(lambda: {'ae': [], 'opt': []})

    for data in dataloader:
        data = data.to(device)

        # Clone and enable gradients on inputs to compute saliency
        # x shape: [total_nodes_in_batch, 2, T]
        x = data.x.clone().detach().requires_grad_(True)
        edge_index = data.edge_index
        batch = data.batch          # node -> graph id
        labels = data.y             # [num_graphs] ground-truth class per graph

        # Forward pass -> logits [num_graphs, num_classes]
        logits = model(x, edge_index, batch)

        # Pick the ground-truth logit for each graph; sum to create a scalar for backprop
        score = logits[torch.arange(len(labels)), labels]
        score.sum().backward()  # compute d(score)/d(x)

        # Saliency magnitude per node/channel/time
        saliency = x.grad.detach().abs()         # [N, 2, T]

        # Channel-wise saliency per node (sum over time dimension)
        ae_saliency = saliency[:, 0, :].sum(dim=1)   # [N]
        opt_saliency = saliency[:, 1, :].sum(dim=1)  # [N]

        # Aggregate node saliency into graph-level saliency via global max pool
        from torch_geometric.nn import global_max_pool
        ae_graphwise = global_max_pool(ae_saliency, batch)   # [num_graphs]
        opt_graphwise = global_max_pool(opt_saliency, batch)  # [num_graphs]

        # Accumulate per class (mean across graphs of the same label later)
        for cls in torch.unique(labels):
            mask = (labels == cls)
            per_class_contrib[cls.item()]['ae'].append(
                ae_graphwise[mask].mean().item())
            per_class_contrib[cls.item()]['opt'].append(
                opt_graphwise[mask].mean().item())

        # Important: clear grads on model to avoid accumulation
        model.zero_grad()

    # Normalize per class to percentages and compute population std
    summary_dict = {}
    for cls, val in per_class_contrib.items():
        ae_tensor = torch.tensor(val['ae'])
        opt_tensor = torch.tensor(val['opt'])

        ae_avg = ae_tensor.mean().item()
        opt_avg = opt_tensor.mean().item()
        ae_std = ae_tensor.std(unbiased=False).item()
        opt_std = opt_tensor.std(unbiased=False).item()

        total = max(ae_avg + opt_avg, 1e-12)  # guard against divide-by-zero

        summary_dict[cls] = {
            'AE %': 100 * ae_avg / total,
            'AE std': 100 * ae_std / total,
            'Optical %': 100 * opt_avg / total,
            'Optical std': 100 * opt_std / total
        }

    # Human-friendly class names for ticks
    class_labels = {
        0: "20%-Cu",
        1: "40%-Cu",
        2: "60%-Cu",
        3: "80%-Cu",
        4: "100%-Cu"
    }

    # ---- Plot configuration ----
    ch1_label = 'Acoustic emission'
    ch2_label = 'Optical emission'
    bar_params = {'width': 0.35, 'alpha': 0.85,
                  'edgecolor': 'k', 'linewidth': 0.5}
    error_params = {'elinewidth': 2, 'capthick': 2,
                    'capsize': 7, 'ecolor': 'black'}
    font_sizes = {'title': 16, 'axis': 16,
                  'ticks': 14, 'legend': 14, 'annotations': 12}

    classes = list(summary_dict.keys())
    ae_vals = [summary_dict[c]['AE %'] for c in classes]
    opt_vals = [summary_dict[c]['Optical %'] for c in classes]
    ae_stds = [summary_dict[c]['AE std'] for c in classes]
    opt_stds = [summary_dict[c]['Optical std'] for c in classes]

    x_pos = range(len(classes))
    width = bar_params['width']

    fig, ax = plt.subplots(figsize=(8, 5))

    # Side-by-side bars with error bars
    ae_bars = ax.bar(x_pos, ae_vals, yerr=ae_stds, label=ch1_label,
                     color='red', **bar_params, error_kw=error_params)
    opt_bars = ax.bar([i + width for i in x_pos], opt_vals, yerr=opt_stds,
                      label=ch2_label, color='blue', **bar_params, error_kw=error_params)

    # Percentage annotations above bars (placed above error bar tips)
    for bar, err in zip(ae_bars, ae_stds):
        height = bar.get_height()
        top = height + err
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, top),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=font_sizes['annotations'])

    for bar, err in zip(opt_bars, opt_stds):
        height = bar.get_height()
        top = height + err
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, top),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=font_sizes['annotations'])

    # Custom x-ticks
    xtick_labels = [class_labels.get(c, str(c)) for c in classes]
    ax.set_xticks([i + width / 2 for i in x_pos])
    ax.set_xticklabels(xtick_labels, fontsize=font_sizes['ticks'])

    # Reference line at 50%
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_ylabel('Channel contribution (%)', fontsize=font_sizes['axis'])
    ax.set_title('AE vs Optical contribution', fontsize=font_sizes['title'])
    ax.set_ylim(0, 110)
    ax.tick_params(axis='y', labelsize=font_sizes['ticks'])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=font_sizes['legend'])

    plt.tight_layout()
    os.makedirs(plot_folder, exist_ok=True)
    save_path = os.path.join(plot_folder, f"{save_prefix}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=350, facecolor='white')
    plt.show()

    return summary_dict


def Node_saliency_per_class(model, correct_graphs, correct_labels, device, plot_folder, save_prefix, ratio=0.5):
    """
    Compute and visualize normalized per-node saliency for each class (balanced subset).

    This function:
    1) Groups graphs by class and selects a balanced subset per class
       (size = floor(min_class_size * ratio), >= 1).
    2) For each selected graph, backpropagates the predicted class score w.r.t. inputs.
    3) Reduces saliency across channels/time -> node saliency, then L1-normalizes
       within each graph so node saliencies sum to 1.
    4) Averages node saliency vectors across graphs in the same class and plots
       a grouped bar chart (node index vs. saliency) for all classes.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model. Must accept (x, edge_index, batch) and return logits [B, C].
    correct_graphs : Sequence[torch_geometric.data.Data]
        Graphs to analyze.
    correct_labels : Sequence[torch.Tensor] or list[int]
        Class labels aligned with `correct_graphs`. Each item is a scalar tensor or int.
    device : torch.device or str
        Device on which to run the computations.
    plot_folder : str
        Directory to save the figure.
    save_prefix : str
        Filename prefix for the saved plot (PNG).
    ratio : float, optional (default=0.5)
        Fraction of the smallest class size to sample for balancing (0 < ratio <= 1).

    Returns
    -------
    avg_saliency_per_class : dict[int, torch.Tensor]
        class_id -> mean node saliency vector [num_nodes].
    count_per_class : dict[int, int]
        class_id -> number of graphs used from that class.

    Notes
    -----
    - Uses the *predicted* class for saliency backprop in each graph
      (i.e., argmax over logits).
      If you prefer GT-based attributions, replace pred with the ground-truth label.
    - Saliency reduction: mean over channels and time dims -> node score.
    - Node saliencies are L1-normalized per graph before averaging.
    """
    model = model.to(device)
    model.eval()

    # ---- 1) Group graphs by class ----
    class_to_graphs = defaultdict(list)
    for graph, label in zip(correct_graphs, correct_labels):
        class_to_graphs[int(label.item())].append(graph)

    # ---- 2) Balanced subset size ----
    min_class_size = min(len(glist) for glist in class_to_graphs.values())
    sample_size = max(1, int(min_class_size * ratio))
    print(f"🟢 Balanced saliency: using {
          sample_size} samples per class (from {min_class_size})")

    # ---- 3) Compute saliency per selected graph ----
    per_class_saliency = defaultdict(list)
    count_per_class = defaultdict(int)

    for cls, graphs in class_to_graphs.items():
        # Deterministic sub-sample (no replacement)
        selected_graphs = resample(
            graphs, replace=False, n_samples=sample_size, random_state=42)

        for graph in selected_graphs:
            graph = graph.to(device)

            # Enable gradients on inputs
            x = graph.x.clone().detach().requires_grad_(True)

            # Forward; logits shape [1, C] since each Data is a single graph
            output = model(x, graph.edge_index, graph.batch)
            # predicted class index
            pred_class = output.argmax(dim=1)

            # Backprop predicted score wrt x (choose the scalar logit)
            output[0, pred_class].backward()

            # Saliency -> average across channels/time to get node scores
            # saliency shape: [num_nodes, 2, T]
            saliency = x.grad.detach().abs()
            node_saliency = saliency.mean(dim=(1, 2))      # [num_nodes]

            # L1-normalize so per-graph node scores sum to 1 (comparable across graphs)
            denom = node_saliency.sum().clamp_min(1e-12)
            node_saliency = node_saliency / denom

            per_class_saliency[cls].append(node_saliency)
            count_per_class[cls] += 1

            # Clear model grads for safety before next iteration
            model.zero_grad()

    # ---- 4) Average node saliency within each class ----
    avg_saliency_per_class = {}
    for cls, saliency_list in per_class_saliency.items():
        # [samples, num_nodes]
        stacked = torch.stack(saliency_list, dim=0)
        avg_saliency = stacked.mean(dim=0)                # [num_nodes]
        avg_saliency_per_class[cls] = avg_saliency

    # ---- 5) Plot grouped bars: node index vs mean saliency for each class ----
    class_labels = {
        0: "20%-Cu",
        1: "40%-Cu",
        2: "60%-Cu",
        3: "80%-Cu",
        4: "100%-Cu"
    }

    font_size = 12
    num_classes = len(avg_saliency_per_class)
    num_nodes = len(next(iter(avg_saliency_per_class.values())))
    node_indices = np.arange(num_nodes)
    bar_width = 0.15

    # Distinct colors per class (tab20 gives many distinct hues)
    all_labels_set = set(avg_saliency_per_class.keys())
    cmap = plt.get_cmap('tab20')
    color_map = {label: cmap(i / len(all_labels_set))
                 for i, label in enumerate(sorted(all_labels_set))}

    # Offsets to place class bars around each node index
    offset = np.linspace(-bar_width*2, bar_width*2, num_classes)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cls in enumerate(sorted(avg_saliency_per_class.keys())):
        saliency = avg_saliency_per_class[cls].cpu().numpy()
        ax.bar(node_indices + offset[i], saliency,
               width=bar_width, label=class_labels[cls],
               color=color_map[cls], alpha=1.0)

    ax.set_xlabel("Node Index (Temporal Segments)", fontsize=font_size)
    ax.set_ylabel("Normalized node saliency", fontsize=font_size)
    ax.set_title("Per-Node normalized saliency", fontsize=font_size + 2)
    ax.set_xticks(node_indices)
    ax.set_xticklabels([str(i + 1)
                       for i in node_indices], fontsize=font_size - 1)

    # Legend outside to keep plot area clean
    ax.legend(
        title="Class",
        fontsize=font_size - 1,
        title_fontsize=font_size,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs(plot_folder, exist_ok=True)
    save_path = os.path.join(plot_folder, f"{save_prefix}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=350, facecolor='white')
    print(f"✅ Saved plot to {save_path}")
    plt.show()

    return avg_saliency_per_class, count_per_class
