# -*- coding: utf-8 -*-
"""
Visualization functions for the multimodal GAT project.

This module provides visualization tools for:
- Raw time series data exploration
- Learned shapelet visualization and comparisons
- Shapelet activation maps per class/channel (balanced subsets)
- Global and class-wise GAT attention heatmaps
- Low-dimensional graph embedding visualizations (e.g., t-SNE)

Notes
-----
- Functions are designed to be "drop-in": signatures preserved.
- Minimal refactors: duplicate imports removed; small f-strings fixed.
- Class label mapping is kept (0..4) -> {"20%-Cu", ..., "100%-Cu"}.

Any reuse of this code should be authorized by the code author.
Developed for the publication:
"Modality-Aware and Light-Weight Graph Attention Networkfor In-SituComposition Monitoring 
in PBF-LB of Graded 316L–CuCrZr Alloys by Sensor Fusion of Optical and Acoustic Emissions"
"""

from typing import Dict, List, Optional, Sequence, Tuple

import os
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import umap  # noqa: F401  # (import kept for potential future use)
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA  # noqa: F401
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: F401
from sklearn.metrics import classification_report, confusion_matrix  # noqa: F401
from sklearn.utils import resample
from torch_geometric.nn import GATConv  # noqa: F401

# -----------------------------------------------------------------------------
# 1) RAW SERIES VISUALIZATION
# -----------------------------------------------------------------------------


def plot_combined_series(
    D1: np.ndarray,
    D2: np.ndarray,
    Y: np.ndarray,
    plot_folder: Optional[str] = None,
    num_samples_per_class: int = 1,
    sampling_rate: float = 400_000,
    figsize: Optional[Tuple[float, float]] = None,
    highlight_regions: Optional[Dict[int, List[Tuple[float, float]]]] = None,
) -> None:
    """
    Plot paired raw time-series (Acoustic vs Back-reflection) for each class.

    For each class in Y, selects up to `num_samples_per_class` random samples,
    and plots Channel-1 (AE) and Channel-2 (Back-reflection) in two columns.
    AE y-limits are uniform across classes; Back-reflection y-limits adapt per row.

    Parameters
    ----------
    D1, D2 : np.ndarray, shape (N, L)
        Channel-1 (AE) and Channel-2 (Back-reflection) arrays.
    Y : np.ndarray, shape (N,)
        Integer class labels (expected 0..4).
    plot_folder : str, optional
        If provided, saves figure to this folder as 'timeseries_comparison.png'.
    num_samples_per_class : int
        Number of traces per class to display (randomly sampled).
    sampling_rate : float
        Samples per second (Hz) for time axis construction.
    figsize : tuple(float, float), optional
        Matplotlib figure size. If None, set based on #classes.
    highlight_regions : dict[int, list[tuple(float, float)]], optional
        Per-class list of (start_time, end_time) spans to highlight.

    Notes
    -----
    - X ticks are set to [0, 2.5, 5, 7.5, 10, 12.5] if they fit the time range.
    - Class name mapping below can be adapted to your dataset.
    """
    if isinstance(num_samples_per_class, str):
        try:
            num_samples_per_class = int(num_samples_per_class)
        except ValueError:
            raise ValueError(
                "num_samples_per_class must be an int or castable string.")

    assert len(D1) == len(D2) == len(Y), "Input arrays must have same length."
    assert num_samples_per_class > 0, "Must plot at least one sample per class."

    N, L = D1.shape
    unique_classes = np.unique(Y)
    num_classes = len(unique_classes)

    if figsize is None:
        figsize = (2 * num_classes, 14)

    # --- Build time axis with adaptive units ---
    time = np.arange(L) / sampling_rate
    if L / sampling_rate < 1e-3:
        time, time_unit = time * 1e6, 'μs'
    elif L / sampling_rate < 1:
        time, time_unit = time * 1e3, 'ms'
    elif L / sampling_rate > 60:
        time, time_unit = time / 60, 'min'
    else:
        time_unit = 's'

    # --- Choose samples per class ---
    selected_indices: List[int] = []
    for cls in unique_classes:
        cls_indices = np.where(Y == cls)[0]
        n_samples = min(len(cls_indices), num_samples_per_class)
        selected_indices.extend(np.random.choice(
            cls_indices, n_samples, replace=False))

    def get_adaptive_limits(data, padding=0.1):
        d_min, d_max = np.min(data), np.max(data)
        span = d_max - d_min if d_max > d_min else 1.0
        return d_min - padding * span, d_max + padding * span

    # AE y-limits uniform across all rows
    y1_min, y1_max = get_adaptive_limits(D1[selected_indices])

    # Color map (consistent per class)
    cmap = plt.get_cmap('tab20')
    all_labels_set = set(unique_classes)
    color_map = {label: cmap(i / len(all_labels_set))
                 for i, label in enumerate(sorted(all_labels_set))}

    # Subplots: rows = classes, cols = 2 channels
    fig, axes = plt.subplots(num_classes, 2, figsize=figsize, sharex=True)
    if num_classes == 1:
        axes = np.array([axes])  # enforce 2D indexing

    # A pretty mapping for titles
    class_labels = {0: "20%-Cu", 1: "40%-Cu",
                    2: "60%-Cu", 3: "80%-Cu", 4: "100%-Cu"}

    for i, cls in enumerate(unique_classes):
        cls_indices = [si for si in selected_indices if Y[si] == cls]
        label_text = class_labels.get(int(cls), f"Class {cls}")

        for sample_idx in cls_indices:
            color = color_map[cls]
            # Channel 1: Acoustic Emission
            axes[i, 0].plot(time, D1[sample_idx], color=color,
                            label=f'Sample {sample_idx}', linewidth=1.2)
            # Channel 2: Back Reflection
            axes[i, 1].plot(time, D2[sample_idx], color=color,
                            linestyle=':', linewidth=1.2)

            # Optional highlighted time regions
            if highlight_regions and cls in highlight_regions:
                for start, end in highlight_regions[cls]:
                    axes[i, 0].axvspan(start, end, color='yellow', alpha=0.3)
                    axes[i, 1].axvspan(start, end, color='yellow', alpha=0.3)

        # Per-axis limits/labels
        for j in range(2):
            if j == 0:
                axes[i, j].set_ylim(y1_min, y1_max)  # uniform for AE
            else:
                y2_row_min, y2_row_max = get_adaptive_limits(D2[cls_indices])
                axes[i, j].set_ylim(y2_row_min, y2_row_max)  # adaptive for BR

            # Optional fixed x-ticks (only keep those within range)
            desired_ticks = np.array([0, 2.5, 5, 7.5, 10, 12.5])
            valid_ticks = desired_ticks[(desired_ticks >= time[0]) & (
                desired_ticks <= time[-1])]
            if len(valid_ticks) >= 2:
                axes[i, j].set_xticks(valid_ticks)

            axes[i, j].tick_params(axis='both', labelsize=12)
            axes[i, j].grid(False)
            if i == num_classes - 1:
                axes[i, j].set_xlabel(f'Time ({time_unit})', fontsize=12)

        # Titles and formatting
        axes[i, 0].set_ylabel('Amplitude', fontsize=12)
        axes[i, 0].set_title(
            f'{label_text} - Acoustic emission', pad=6, fontsize=16)
        axes[i, 1].set_title(
            f'{label_text} - Optical emission', pad=6, fontsize=16)

        # Add black border for Back Reflection plots (right column)
        for spine in axes[i, 1].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

    # Global title and layout
    fig.suptitle(f'Raw time-series data\nSampling Rate: {sampling_rate / 1000:.1f} kHz',
                 y=0.995, fontsize=16, fontweight='light')
    fig.subplots_adjust(top=0.93, left=0.08, right=0.98,
                        hspace=0.4, wspace=0.15)

    # Save
    if plot_folder:
        os.makedirs(plot_folder, exist_ok=True)
        save_path = os.path.join(plot_folder, 'timeseries_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'[✔] Saved visualization to {save_path}')

    plt.show()
    plt.close()


# -----------------------------------------------------------------------------
# 2) SHAPELET VISUALIZATIONS
# -----------------------------------------------------------------------------

def visualize_shapelets(
    shapelets_ch1,
    shapelets_ch2,
    plot_folder: Optional[str] = None,
    save_name: str = "Learned Shapelets",
) -> None:
    """
    Visualize learned shapelets for both channels side-by-side.

    Parameters
    ----------
    shapelets_ch1, shapelets_ch2 : (Tensor | np.ndarray), shape (K, L)
        Shapelet banks for Channel-1 and Channel-2 (assumed same K, L).
    plot_folder : str, optional
        If provided, saves figure as <save_name>.png inside this folder.
    save_name : str
        Base filename for the saved plot (without extension).
    """
    # Convert tensors to numpy
    if isinstance(shapelets_ch1, torch.Tensor):
        shapelets_ch1 = shapelets_ch1.detach().cpu().numpy()
    if isinstance(shapelets_ch2, torch.Tensor):
        shapelets_ch2 = shapelets_ch2.detach().cpu().numpy()

    num_shapelets = shapelets_ch1.shape[0]

    # Style configuration
    ch1_color, ch2_color = '#FF0000', '#000080'  # red/navy
    linewidth = 2.5
    font_sizes = {'title': 12, 'label': 12, 'tick': 12}

    # Global y-limits/ticks across channels
    global_min = min(np.min(shapelets_ch1), np.min(shapelets_ch2))
    global_max = max(np.max(shapelets_ch1), np.max(shapelets_ch2))
    y_padding = (global_max - global_min) * 0.1
    y_lim = (global_min - y_padding, global_max + y_padding)
    y_ticks = np.unique(
        np.round(np.linspace(y_lim[0], y_lim[1], 3)).astype(int))

    # Create subplots: rows = shapelets, cols = channels
    fig, axes = plt.subplots(num_shapelets, 2, figsize=(
        8, 12), sharex=True, sharey=False)
    if num_shapelets == 1:
        axes = axes.reshape(1, 2)

    for i in range(num_shapelets):
        # Channel 1
        ax1 = axes[i, 0]
        ax1.plot(shapelets_ch1[i], color=ch1_color,
                 linewidth=linewidth, alpha=0.8)
        ax1.set_title(
            f"Channel 1 - Shapelet {i+1}", pad=5, fontsize=font_sizes['title'])
        ax1.set_ylabel("Amplitude", fontsize=font_sizes['label'])
        ax1.set_ylim(y_lim)
        ax1.set_yticks(y_ticks)
        ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
        ax1.tick_params(axis='both', labelsize=font_sizes['tick'])

        # Channel 2
        ax2 = axes[i, 1]
        ax2.plot(shapelets_ch2[i], color=ch2_color,
                 linewidth=linewidth, alpha=0.8)
        ax2.set_title(
            f"Channel 2 - Shapelet {i+1}", pad=5, fontsize=font_sizes['title'])
        ax2.set_ylabel("Amplitude", fontsize=font_sizes['label'])
        ax2.set_ylim(y_lim)
        ax2.set_yticks(y_ticks)
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
        ax2.tick_params(axis='both', labelsize=font_sizes['tick'])

        # Bottom x-axis labels
        if i == num_shapelets - 1:
            for ax in [ax1, ax2]:
                ax.set_xlabel("Datapoints", fontsize=font_sizes['label'])
                ax.xaxis.set_major_locator(plt.MaxNLocator(5, integer=True))

    plt.tight_layout()
    fig.suptitle(f"{save_name}", y=0.995, fontsize=16, fontweight='light')
    fig.subplots_adjust(top=0.93, left=0.08, right=0.98,
                        hspace=0.4, wspace=0.15)

    # Save
    if plot_folder:
        os.makedirs(plot_folder, exist_ok=True)
        out_path = os.path.join(plot_folder, f"{save_name}.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"✅ Saved high-res plot to: {out_path}")

    plt.show()
    plt.close()


def plot_shapelets_side_by_side(
    ch1_init,
    ch1_trained,
    ch2_init,
    ch2_trained,
    save_name: str = "compare_shapelets_overlayed",
    plot_folder: Optional[str] = None,
) -> None:
    """
    Overlay initial vs trained shapelets for both channels.

    Initial = dotted grey; Trained = solid red/navy.
    Keeps global y-limits consistent across channels/shapelets.

    Parameters
    ----------
    ch1_init, ch1_trained, ch2_init, ch2_trained : (Tensor | np.ndarray), (K, L)
        Shapelet banks before/after training.
    save_name : str
        Output base filename (without extension).
    plot_folder : str, optional
        If provided, saves figure to this folder.
    """
    # Convert to numpy
    def to_np(x): return x.detach().cpu().numpy(
    ) if isinstance(x, torch.Tensor) else x
    ch1_init, ch1_trained = to_np(ch1_init), to_np(ch1_trained)
    ch2_init, ch2_trained = to_np(ch2_init), to_np(ch2_trained)

    num_shapelets, L = ch1_init.shape

    font_sizes = {'title': 13, 'label': 12, 'tick': 11}
    linewidth = 1.5

    # Global y-limits/ticks
    global_min = min(np.min(ch1_init), np.min(ch1_trained),
                     np.min(ch2_init), np.min(ch2_trained))
    global_max = max(np.max(ch1_init), np.max(ch1_trained),
                     np.max(ch2_init), np.max(ch2_trained))
    y_padding = (global_max - global_min) * 0.1
    y_lim = (global_min - y_padding, global_max + y_padding)
    y_ticks = np.round(np.linspace(y_lim[0], y_lim[1], 3)).tolist()

    fig, axes = plt.subplots(num_shapelets, 2, figsize=(
        8, 12), sharex=True, sharey=False)
    if num_shapelets == 1:
        axes = axes.reshape(1, 2)

    for i in range(num_shapelets):
        # Channel 1
        ax1 = axes[i, 0]
        ax1.plot(ch1_trained[i], color='red',
                 linewidth=linewidth, label='Trained')
        ax1.plot(ch1_init[i], linestyle=':',
                 color='black', linewidth=2, label='Init')
        ax1.set_title(
            f"Acoustic emission - Shapelet {i+1}", fontsize=font_sizes['title'], pad=4)
        ax1.set_ylabel("Amplitude", fontsize=font_sizes['label'])
        ax1.set_ylim(y_lim)
        ax1.set_yticks(y_ticks)
        ax1.tick_params(axis='both', labelsize=font_sizes['tick'])
        ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%d'))

        # Channel 2
        ax2 = axes[i, 1]
        ax2.plot(ch2_trained[i], color='navy',
                 linewidth=linewidth, label='Trained')
        ax2.plot(ch2_init[i], linestyle=':',
                 color='black', linewidth=2, label='Init')
        ax2.set_title(
            f"Optical emission - Shapelet {i+1}", fontsize=font_sizes['title'], pad=4)
        ax2.set_ylim(y_lim)
        ax2.set_yticks(y_ticks)
        ax2.tick_params(axis='both', labelsize=font_sizes['tick'])
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%d'))

        if i == num_shapelets - 1:
            ax1.set_xlabel("Datapoints", fontsize=font_sizes['label'])
            ax2.set_xlabel("Datapoints", fontsize=font_sizes['label'])
            ax1.xaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
            ax2.xaxis.set_major_locator(plt.MaxNLocator(5, integer=True))

    # Global title and legend
    fig.suptitle("Overlayed Shapelet Evolution (Initial vs Trained)",
                 fontsize=16, y=0.995, fontweight='light')
    fig.subplots_adjust(top=0.94, left=0.08, right=0.98,
                        hspace=0.4, wspace=0.15)

    legend_elements = [
        Line2D([0], [0], color='grey', linestyle='--',
               linewidth=1.5, label='Initial'),
        Line2D([0], [0], color='red', linestyle='-',
               linewidth=2.5, label='Trained (Acoustic emission)'),
        Line2D([0], [0], color='navy', linestyle='-',
               linewidth=2.5, label='Trained (Optical emission)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               fontsize=12, ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.03))

    if plot_folder:
        os.makedirs(plot_folder, exist_ok=True)
        out_path = os.path.join(plot_folder, f"{save_name}.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"✅ Saved high-res plot to: {out_path}")

    plt.show()
    plt.close()


# -----------------------------------------------------------------------------
# 3) SHAPELET ACTIVATION MAPS (PER CLASS / STACKED / ALL)
# -----------------------------------------------------------------------------

def visualize_classwise_node_activations_per_channel(
    correct_graphs,
    correct_labels,
    shapelet_model,
    plot_folder: str,
    use_mean: bool = False,
    ratio: float = 0.5,
) -> None:
    """
    For each class, compute a balanced subset of correctly predicted graphs,
    and visualize the average node-wise shapelet activations per channel.

    Each shapelet is min-max normalized across nodes (per sample) prior to averaging.

    Parameters
    ----------
    correct_graphs : list[Data]
        Correctly classified graphs.
    correct_labels : torch.Tensor | list[int]
        Ground-truth labels aligned with correct_graphs.
    shapelet_model : torch.nn.Module
        Model exposing `forward(x)` or `forward_get_channelwise(x)`.
        Assumes dual-channel outputs if using the combined `forward`.
    plot_folder : str
        Output directory for the saved figures.
    use_mean : bool
        If True, calls `forward_get_channelwise(x)`; else splits `forward(x)` into two halves.
    ratio : float
        Fraction of min per-class size to sample (0 < ratio <= 1).
    """
    os.makedirs(plot_folder, exist_ok=True)
    font_size = 16
    shapelet_model.eval()

    # Group graphs by class
    class_to_graphs = defaultdict(list)
    for graph, label in zip(correct_graphs, correct_labels):
        class_to_graphs[int(label.item())].append(graph)

    # Balanced subset size
    min_class_size = min(len(glist) for glist in class_to_graphs.values())
    sample_size = max(1, int(min_class_size * ratio))
    print(f"🟢 Balanced saliency: using {
          sample_size} samples per class (from {min_class_size})")

    class_labels = {0: "20%-Cu", 1: "40%-Cu",
                    2: "60%-Cu", 3: "80%-Cu", 4: "100%-Cu"}

    # Per class processing
    for class_label, graphs in class_to_graphs.items():
        selected = resample(graphs, replace=False,
                            n_samples=sample_size, random_state=42)

        all_ch1, all_ch2 = [], []

        for graph in selected:
            with torch.no_grad():
                x = graph.x.to(shapelet_model.shapelets_ch1.device)

                if use_mean:
                    d1, d2 = shapelet_model.forward_get_channelwise(x)
                    d1, d2 = d1.cpu().numpy(), d2.cpu().numpy()
                else:
                    dist = shapelet_model(x).cpu().numpy()
                    K = dist.shape[1] // 2
                    d1, d2 = dist[:, :K], dist[:, K:]

                # Min-max per shapelet across nodes
                d1 = (d1 - d1.min(axis=0, keepdims=True)) / (d1.max(axis=0,
                                                                    keepdims=True) - d1.min(axis=0, keepdims=True) + 1e-8)
                d2 = (d2 - d2.min(axis=0, keepdims=True)) / (d2.max(axis=0,
                                                                    keepdims=True) - d2.min(axis=0, keepdims=True) + 1e-8)

                all_ch1.append(d1)
                all_ch2.append(d2)

        # Align by smallest node count
        min_nodes = min(d.shape[0] for d in all_ch1)
        ch1_avg = np.mean([d[:min_nodes, :] for d in all_ch1], axis=0)
        ch2_avg = np.mean([d[:min_nodes, :] for d in all_ch2], axis=0)

        # Plotting
        fig = plt.figure(figsize=(16, 7))
        gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1], hspace=0.25)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        cbar_ax = fig.add_subplot(gs[1, :])

        vmin, vmax = 0, 1
        im1 = ax1.imshow(ch1_avg.T, cmap='PRGn',
                         aspect='auto', vmin=vmin, vmax=vmax)
        im2 = ax2.imshow(ch2_avg.T, cmap='PRGn',
                         aspect='auto', vmin=vmin, vmax=vmax)

        for ax in [ax1, ax2]:
            ax.set_xlabel("Node index", fontsize=font_size + 2)
            ax.set_ylabel("Shapelet index", fontsize=font_size + 2)
            ax.set_xticks(np.arange(min_nodes))
            ax.set_xticklabels(np.arange(1, min_nodes + 1), fontsize=font_size)
            ax.set_yticks(np.arange(ch1_avg.shape[1]))
            ax.set_yticklabels(
                np.arange(1, ch1_avg.shape[1] + 1), fontsize=font_size)

        ax1.set_title("Shapelet activation map (Acoustic emission)",
                      fontsize=font_size + 2)
        ax2.set_title("Shapelet activation map (Optical emission)",
                      fontsize=font_size + 2)

        cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Normalized distance metric", fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)

        class_name = class_labels.get(class_label, f"Class {class_label}")
        plt.suptitle(f"{class_name}: Node-wise shapelet activation",
                     fontsize=font_size + 4, y=1.05)

        save_path = os.path.join(plot_folder, f"class_{
                                 class_label}_node_activation_channels.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved: {save_path}")
        plt.show()
        plt.close()


def visualize_classwise_node_activations_stacked(
    correct_graphs,
    correct_labels,
    shapelet_model,
    plot_folder: str,
    use_mean: bool = False,
    ratio: float = 0.5,
) -> None:
    """
    Stacked 5×2 grid of average node-wise shapelet activations per class.

    For each class row: left = AE heatmap, right = Back-reflection heatmap.
    Each shapelet activation is min-max normalized across nodes (per sample),
    then averaged across the balanced subset.

    Parameters
    ----------
    correct_graphs : list[Data]
        Correctly classified graphs.
    correct_labels : torch.Tensor | list[int]
        Ground-truth labels aligned with correct_graphs.
    shapelet_model : torch.nn.Module
        Shapelet model with `forward` or `forward_get_channelwise`.
    plot_folder : str
        Directory to save the stacked figure.
    use_mean : bool
        If True, uses `forward_get_channelwise`; else splits `forward`.
    ratio : float
        Fraction of min per-class size to sample.
    """
    font_size = 16
    shapelet_model.eval()

    class_labels = {0: "20%-Cu", 1: "40%-Cu",
                    2: "60%-Cu", 3: "80%-Cu", 4: "100%-Cu"}

    # Group by class
    class_to_graphs = defaultdict(list)
    for graph, label in zip(correct_graphs, correct_labels):
        class_to_graphs[int(label.item())].append(graph)

    # Balanced subset
    min_class_size = min(len(glist) for glist in class_to_graphs.values())
    sample_size = max(1, int(min_class_size * ratio))
    print(f"🟢 Balanced saliency: using {
          sample_size} samples per class (from {min_class_size})")

    num_classes = len(class_to_graphs)
    fig, axes = plt.subplots(num_classes, 2, figsize=(
        14, 3.8 * num_classes), sharey='row')
    plt.subplots_adjust(hspace=0.6)
    vmin, vmax = 0, 1

    for row, class_label in enumerate(sorted(class_to_graphs.keys())):
        selected = resample(
            class_to_graphs[class_label], replace=False, n_samples=sample_size, random_state=42)

        all_ch1, all_ch2 = [], []
        for graph in selected:
            with torch.no_grad():
                x = graph.x.to(shapelet_model.shapelets_ch1.device)
                if use_mean:
                    d1, d2 = shapelet_model.forward_get_channelwise(x)
                    d1, d2 = d1.cpu().numpy(), d2.cpu().numpy()
                else:
                    dist = shapelet_model(x).cpu().numpy()
                    K = dist.shape[1] // 2
                    d1, d2 = dist[:, :K], dist[:, K:]

                # Min-max per shapelet across nodes
                d1 = (d1 - d1.min(axis=0, keepdims=True)) / (d1.max(axis=0,
                                                                    keepdims=True) - d1.min(axis=0, keepdims=True) + 1e-8)
                d2 = (d2 - d2.min(axis=0, keepdims=True)) / (d2.max(axis=0,
                                                                    keepdims=True) - d2.min(axis=0, keepdims=True) + 1e-8)

                all_ch1.append(d1)
                all_ch2.append(d2)

        min_nodes = min(d.shape[0] for d in all_ch1)
        ch1_avg = np.mean([d[:min_nodes, :] for d in all_ch1], axis=0)
        ch2_avg = np.mean([d[:min_nodes, :] for d in all_ch2], axis=0)

        ax1 = axes[row, 0]
        ax2 = axes[row, 1]
        im1 = ax1.imshow(ch1_avg.T, cmap='PRGn',
                         aspect='auto', vmin=vmin, vmax=vmax)
        im2 = ax2.imshow(ch2_avg.T, cmap='PRGn',
                         aspect='auto', vmin=vmin, vmax=vmax)

        for ax in [ax1, ax2]:
            ax.set_xlabel("Node index", fontsize=font_size + 2)
            if ax is ax1:
                ax.set_ylabel("Shapelet index", fontsize=font_size + 3)
            ax.set_xticks(np.arange(min_nodes))
            ax.set_xticklabels(np.arange(1, min_nodes + 1),
                               fontsize=font_size - 1)
            ax.set_yticks(np.arange(ch1_avg.shape[1]))
            ax.set_yticklabels(
                np.arange(1, ch1_avg.shape[1] + 1), fontsize=font_size - 1)
            ax.tick_params(axis='both', which='major', labelsize=font_size - 1)

        ax1.set_title("Acoustic emission", fontsize=font_size + 2)
        ax2.set_title("Optical emission", fontsize=font_size + 2)

    # Add centered class titles above each row
    for row, class_label in enumerate(sorted(class_to_graphs.keys())):
        pos_left = axes[row, 0].get_position()
        pos_right = axes[row, 1].get_position()
        x_center = (pos_left.x0 + pos_right.x1) / 2
        y_top = max(pos_left.y1, pos_right.y1) + 0.015
        plt.gcf().text(x_center, y_top, f"{class_labels[class_label]}", ha='center', va='bottom',
                       fontsize=font_size + 4, weight='bold')

    # Shared colorbar
    cbar_ax = plt.gcf().add_axes([0.25, 0.05, 0.5, 0.02])
    cbar = plt.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Normalized distance metric", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size - 1)

    # Save
    os.makedirs(plot_folder, exist_ok=True)
    save_path = os.path.join(
        plot_folder, "Stacked_classwise_node_activation_channels.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {save_path}")
    plt.show()


def visualize_node_activations_from_multiple_graphs(
    correct_graphs,
    correct_preds,   # noqa: ARG002 kept for signature parity; not used here
    correct_labels,
    shapelet_model,
    plot_folder: str,
    use_mean: bool = False,
    ratio: float = 0.5,
) -> None:
    """
    Average normalized node-wise shapelet activations per channel across ALL classes.

    Steps:
    - Balanced sampling per class (ratio of min class size).
    - Min-max normalize each shapelet across nodes (per sample).
    - Truncate to smallest node count across samples; average across the pool.
    - Plot AE and Back-reflection activation maps.

    Parameters
    ----------
    correct_graphs : list[Data]
    correct_preds : list[int]
        Unused here (kept to match your original signature).
    correct_labels : list[int] | torch.Tensor
    shapelet_model : torch.nn.Module
    plot_folder : str
    use_mean : bool
    ratio : float
    """
    os.makedirs(plot_folder, exist_ok=True)
    shapelet_model.eval()
    font_size = 16

    # Group by class
    class_to_graphs = defaultdict(list)
    for graph, label in zip(correct_graphs, correct_labels):
        class_to_graphs[int(label.item())].append(graph)

    # Balanced subset size
    min_class_size = min(len(glist) for glist in class_to_graphs.values())
    sample_size = max(1, int(min_class_size * ratio))
    print(f"🟢 Using {sample_size} samples per class from {
          min_class_size} (balanced)")

    all_ch1, all_ch2 = [], []
    for graphs in class_to_graphs.values():
        selected_graphs = resample(
            graphs, replace=False, n_samples=sample_size, random_state=42)
        for graph in selected_graphs:
            with torch.no_grad():
                x = graph.x.to(shapelet_model.shapelets_ch1.device)
                if use_mean:
                    d1, d2 = shapelet_model.forward_get_channelwise(x)
                    d1, d2 = d1.cpu().numpy(), d2.cpu().numpy()
                else:
                    dist = shapelet_model(x).cpu().numpy()
                    K = dist.shape[1] // 2
                    d1, d2 = dist[:, :K], dist[:, K:]

                # Min-max per shapelet across nodes
                d1 = (d1 - d1.min(axis=0, keepdims=True)) / (d1.max(axis=0,
                                                                    keepdims=True) - d1.min(axis=0, keepdims=True) + 1e-8)
                d2 = (d2 - d2.min(axis=0, keepdims=True)) / (d2.max(axis=0,
                                                                    keepdims=True) - d2.min(axis=0, keepdims=True) + 1e-8)

                all_ch1.append(d1)
                all_ch2.append(d2)

    # Truncate to min #nodes and average
    min_nodes = min(d.shape[0] for d in all_ch1)
    ch1_avg = np.mean([d[:min_nodes, :] for d in all_ch1], axis=0)
    ch2_avg = np.mean([d[:min_nodes, :] for d in all_ch2], axis=0)

    # Plot
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1], hspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[1, :])

    vmin, vmax = 0, 1
    im1 = ax1.imshow(ch1_avg.T, cmap='PRGn',
                     aspect='auto', vmin=vmin, vmax=vmax)
    im2 = ax2.imshow(ch2_avg.T, cmap='PRGn',
                     aspect='auto', vmin=vmin, vmax=vmax)

    for ax in [ax1, ax2]:
        ax.set_xlabel("Node index", fontsize=font_size)
        ax.set_ylabel("Shapelet index", fontsize=font_size)
        ax.set_xticks(np.arange(min_nodes))
        ax.set_xticklabels(np.arange(1, min_nodes + 1), fontsize=font_size)
        ax.set_yticks(np.arange(ch1_avg.shape[1]))
        ax.set_yticklabels(
            np.arange(1, ch1_avg.shape[1] + 1), fontsize=font_size)

    ax1.set_title("Shapelet activation map (Acoustic emission)",
                  fontsize=font_size + 2)
    ax2.set_title("Shapelet activation map (Optical emission)",
                  fontsize=font_size + 2)

    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Normalized activation (per shapelet)", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)

    plt.suptitle("Combined node-wise shapelet activation (All classes)",
                 fontsize=font_size + 4, y=1.05)
    save_path = os.path.join(
        plot_folder, "combined_node_activation_all_classes.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()


# -----------------------------------------------------------------------------
# 4) ATTENTION HEATMAPS
# -----------------------------------------------------------------------------

def plot_global_attention_heatmap(
    model,
    correct_graphs,
    device: torch.device,
    save_folder: str,
    normalize_rows: bool = True,
    top_k: Optional[int] = None,
    title: str = "Global GAT Attention",
):
    """
    Aggregate attention across the 3 GAT layers and across multiple graphs,
    then visualize a single global node-to-node attention heatmap.

    Parameters
    ----------
    model : nn.Module
        Must expose `shapelet_model` and three GAT layers as `gat1/gat2/gat3`.
    correct_graphs : list[Data]
        Graphs to summarize over.
    device : torch.device
        Inference device.
    save_folder : str
        Output directory for PNG.
    normalize_rows : bool
        If True, row-normalize attention and scale to [0,1].
    top_k : int, optional
        If provided, keep only top-k outgoing edges per row before normalization.
    title : str
        Title for the heatmap.

    Returns
    -------
    torch.Tensor
        The averaged attention matrix (possibly row-normalized and masked).
    """
    model.eval()
    os.makedirs(save_folder, exist_ok=True)
    A_sum = None

    with torch.no_grad():
        for graph in correct_graphs:
            x_input = model.shapelet_model(graph.x.to(device))
            edge_index = graph.edge_index.to(device)
            num_nodes = x_input.size(0)
            A_global = torch.zeros((num_nodes, num_nodes))

            # Accumulate attention layer-wise
            for gat_layer in [model.gat1, model.gat2, model.gat3]:
                x_input, attn_data = gat_layer(
                    x_input, edge_index, return_attention_weights=True)
                edge_idx = attn_data[0].cpu()                 # [2, E]
                attn_weights = attn_data[1].cpu().mean(
                    dim=1)  # [E] avg across heads

                A_layer = torch.zeros((num_nodes, num_nodes))
                for idx in range(edge_idx.shape[1]):
                    src = edge_idx[0, idx].item()
                    tgt = edge_idx[1, idx].item()
                    A_layer[src, tgt] = attn_weights[idx].item()

                A_global += A_layer
                x_input = F.elu(x_input)

            A_global /= 3.0  # average over layers

            A_sum = A_global.clone() if A_sum is None else (A_sum + A_global)

        A_avg = A_sum / len(correct_graphs)

        # Optional sparsification to top-k per row
        if top_k is not None:
            topk_mask = torch.zeros_like(A_avg)
            for i in range(A_avg.shape[0]):
                _, indices = torch.topk(A_avg[i], k=min(top_k, A_avg.shape[1]))
                topk_mask[i, indices] = 1.0
            A_avg *= topk_mask
            A_avg[A_avg < 1e-6] = 0.0

        # Optional row normalization to [0,1]
        if normalize_rows:
            row_sums = A_avg.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1e-8
            A_avg = A_avg / row_sums
            A_min, A_max = A_avg.min(), A_avg.max()
            A_avg = (A_avg - A_min) / (A_max - A_min + 1e-8)

        # Plot
        A_avg.fill_diagonal_(0)
        plt.figure(figsize=(8, 7))
        ax = sns.heatmap(
            A_avg.numpy(),
            cmap="viridis",
            square=True,
            xticklabels=[str(i + 1) for i in range(A_avg.shape[1])],
            yticklabels=[str(i + 1) for i in range(A_avg.shape[0])],
        )
        plt.title(title)
        plt.xlabel("To Node (j)", fontsize=20)
        plt.ylabel("From Node (i)", fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # Colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_label("Attention strength (Normalized)", fontsize=20)
        cbar.ax.tick_params(labelsize=16)

        plt.tight_layout()

        filename = "Global_attention_heatmap_multiple_graphs"
        if normalize_rows:
            filename += "_rownorm"
        if top_k:
            filename += f"_top{top_k}"
        filepath = os.path.join(save_folder, filename + ".png")
        plt.savefig(filepath, dpi=350, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved: {filepath}")
        plt.show()
        plt.close()

        return A_avg


def plot_classwise_attention_maps(
    model,
    correct_graphs,
    correct_labels,
    device: torch.device,
    save_folder: str,
    ratio: float = 0.02,
) -> None:
    """
    Compute class-wise node-to-node attention maps (averaged over a balanced subset),
    and draw a heatmap per class.

    Parameters
    ----------
    model : nn.Module
        Must expose `shapelet_model` and three GAT layers `gat1/gat2/gat3`.
    correct_graphs : list[Data]
        Correctly classified graphs.
    correct_labels : list[int] | torch.Tensor
        Corresponding labels aligned with `correct_graphs`.
    device : torch.device
        Inference device.
    save_folder : str
        Directory to save class-wise heatmaps.
    ratio : float
        Fraction of min per-class sample count to use (0 < ratio <= 1).
    """
    os.makedirs(save_folder, exist_ok=True)
    model.eval()

    class_labels = {0: "20%-Cu", 1: "40%-Cu",
                    2: "60%-Cu", 3: "80%-Cu", 4: "100%-Cu"}

    # Group graphs by class
    class_data = defaultdict(list)
    for graph, label in zip(correct_graphs, correct_labels):
        class_label = int(label.item()) if torch.is_tensor(
            label) else int(label)
        class_data[class_label].append(graph)

    min_samples = min(len(glist) for glist in class_data.values())
    sample_size = max(1, int(min_samples * ratio))

    for class_label, graph_list in class_data.items():
        selected_graphs = resample(
            graph_list, replace=False, n_samples=sample_size, random_state=42)
        attn_matrices = []

        for graph in selected_graphs:
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            num_nodes = x.size(0)

            with torch.no_grad():
                x = model.shapelet_model(x)
                A_total = torch.zeros((num_nodes, num_nodes))

                for gat_layer in [model.gat1, model.gat2, model.gat3]:
                    x, (ei, alpha) = gat_layer(
                        x, edge_index, return_attention_weights=True)
                    ei = ei.cpu()                 # [2, E]
                    alpha = alpha.cpu().mean(dim=1)  # [E] avg across heads

                    A_layer = torch.zeros((num_nodes, num_nodes))
                    for idx in range(ei.shape[1]):
                        src = ei[0, idx].item()
                        tgt = ei[1, idx].item()
                        A_layer[src, tgt] = alpha[idx].item()

                    A_total += A_layer
                    x = F.elu(x)

                A_total /= 3.0
                attn_matrices.append(A_total)

        # Average across selected graphs
        A_avg = sum(attn_matrices) / len(attn_matrices)

        # Row-normalize and scale to [0,1]
        row_sums = A_avg.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1e-8
        A_avg = A_avg / row_sums
        A_min, A_max = A_avg.min(), A_avg.max()
        A_avg = (A_avg - A_min) / (A_max - A_min + 1e-8)
        A_avg.fill_diagonal_(0)  # remove self-attention for readability

        # Plot
        plt.figure(figsize=(8, 7))
        ax = sns.heatmap(
            A_avg.numpy(),
            cmap="viridis",
            square=True,
            xticklabels=[str(i + 1) for i in range(A_avg.shape[1])],
            yticklabels=[str(i + 1) for i in range(A_avg.shape[0])],
        )

        plt.title(f"Attention Strength — {
                  class_labels[class_label]}", fontsize=22)
        plt.xlabel("To Node (j)", fontsize=20)
        plt.ylabel("From Node (i)", fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        cbar = ax.collections[0].colorbar
        cbar.set_label("Attention strength (Normalized)", fontsize=20)
        cbar.ax.tick_params(labelsize=16)

        plt.tight_layout()
        path = os.path.join(save_folder, f"class_{
                            class_label}_attention_strength.png")
        plt.savefig(path, dpi=350, bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close()
        print(f"✅ Saved: {path}")


# -----------------------------------------------------------------------------
# 5) EMBEDDING VISUALIZATION
# -----------------------------------------------------------------------------

def visualize_embeddings_all_methods(
    model,
    graph_list,
    device: torch.device,
    class_labels: Optional[Dict[int, str]] = None,
    save_folder: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> None:
    """
    Visualize graph-level embeddings (from `model.forward_embedding`) with t-SNE.

    Currently renders only the "Both Channels" mode and t-SNE reducer for simplicity,
    matching your latest code state. It is easy to expand to PCA/UMAP or per-channel
    masking by adding to `reduction_methods` and `modes`.

    Parameters
    ----------
    model : nn.Module
        Must expose `forward_embedding(x, edge_index, batch, mask_channel=None)`.
    graph_list : list[Data]
        Graphs to embed (e.g., test set).
    device : torch.device
        Inference device.
    class_labels : dict[int, str], optional
        Pretty names for classes (defaults to 316L-Cu mapping).
    save_folder : str, optional
        Directory to save figures (PNG).
    figsize : tuple(float, float)
        Matplotlib figure size.

    Notes
    -----
    - Colors/markers are fixed per class for consistency across methods/modes.
    """
    model = model.eval().to(device)

    if class_labels is None:
        class_labels = {0: "20%-Cu", 1: "40%-Cu",
                        2: "60%-Cu", 3: "80%-Cu", 4: "100%-Cu"}

    modes = {"Both Channels": None}
    reduction_methods = {
        "t-SNE": TSNE(n_components=2, perplexity=30, random_state=42)}

    # Colors and markers per class
    all_labels_set = set([g.y.item() for g in graph_list])
    cmap = plt.get_cmap('tab20')
    color_map = {label: cmap(i / len(all_labels_set))
                 for i, label in enumerate(sorted(all_labels_set))}
    markers = ['o', 's', '^', 'D', 'p']  # five classes
    marker_map = {label: markers[i % len(markers)]
                  for i, label in enumerate(sorted(all_labels_set))}

    for mode_name, mask_channel in modes.items():
        embeddings = []
        labels = []

        with torch.no_grad():
            for graph in graph_list:
                graph = graph.to(device)
                # When using single-graph Data, build batch of zeros
                batch = torch.zeros(
                    graph.num_nodes, dtype=torch.long, device=device)
                emb = model.forward_embedding(
                    graph.x, graph.edge_index, batch, mask_channel)
                embeddings.append(emb.cpu().numpy())
                labels.append(graph.y.cpu().item())

        embeddings = np.vstack(embeddings)
        labels = np.array(labels)

        for method_name, reducer in reduction_methods.items():
            reduced = reducer.fit_transform(embeddings)

            plt.figure(figsize=figsize)

            # Legend handles first (stable ordering)
            legend_elements = [
                plt.Line2D([0], [0],
                           marker=marker_map[label],
                           color='w',
                           label=class_labels[label],
                           markerfacecolor=color_map[label],
                           markersize=12,
                           markeredgecolor='k')
                for label in sorted(np.unique(labels))
            ]

            # Plot points by class
            for label in np.unique(labels):
                mask = labels == label
                plt.scatter(
                    reduced[mask, 0],
                    reduced[mask, 1],
                    c=[color_map[label]],
                    marker=marker_map[label],
                    s=75,
                    alpha=0.85,
                    edgecolor='k',
                    linewidths=0.5,
                )

            plt.legend(
                handles=legend_elements,
                title="316L- Cu Composition",
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                fontsize=14,
                title_fontsize=14,
                framealpha=1,
            )

            plt.title(f"Embedding via {method_name} | {
                      mode_name}", fontsize=16, pad=20)
            plt.xlabel("Dimension 1", fontsize=14)
            plt.ylabel("Dimension 2", fontsize=14)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.grid(False)
            plt.tight_layout()

            if save_folder:
                os.makedirs(save_folder, exist_ok=True)
                filename = f"{method_name.lower()}_{mode_name.replace(
                    ' ', '_').lower()}_embeddings.png"
                out_path = os.path.join(save_folder, filename)
                plt.savefig(out_path, dpi=350,
                            bbox_inches='tight', facecolor='white')
                print(f"Saved to {out_path}")

            plt.show()
