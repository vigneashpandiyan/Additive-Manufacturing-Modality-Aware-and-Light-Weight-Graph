# -*- coding: utf-8 -*-
"""Network architectures for multimodal time series classification.

This module implements neural network components for processing dual-channel time series data
using shapelet learning and graph attention mechanisms. The architecture consists of:

1. BatchedShapeletExtractor - Learns discriminative subsequence patterns from each channel
2. GNNWithAttention - Graph neural network with attention for processing shapelet representations

Key Features:
- Joint learning of temporal patterns and graph relationships
- Modality-specific shapelet extraction
- Attention-based feature aggregation
- End-to-end differentiable architecture

Example Usage:
    >>> # Initialize model
    >>> model = GNNWithAttention(in_channels=None, 
    ...                         hidden_channels=32,
    ...                         out_channels=5,
    ...                         shapelet_len=50,
    ...                         num_shapelets=10)
    >>> # Process sample data
    >>> x = torch.randn(128, 2, 300)  # 128 nodes, 2 channels, 300 timesteps
    >>> edge_index = torch.randint(0, 128, (2, 256))  # Random edges
    >>> batch = torch.randint(0, 4, (128,))  # 4 graphs in batch
    >>> output = model(x, edge_index, batch)  # Forward pass


Any reuse of this code should be authorized by the code author.
Developed for the publication:
"Modality-Aware and Light-Weight Graph Attention Networkfor In-SituComposition Monitoring 
in PBF-LB of Graded 316L–CuCrZr Alloys by Sensor Fusion of Optical and Acoustic Emissions"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import matplotlib.pyplot as plt
import numpy as np


class BatchedShapeletExtractor(nn.Module):
    """
    Learns and extracts shapelet features from dual-channel time series data.

    Shapelets are discriminative subsequences learned directly from the data.
    For each input channel, the module:
    1. Maintains a set of learnable shapelets
    2. Computes minimum squared distances between shapelets and all subsequences
    3. Returns concatenated features from both channels

    Architecture:
    - Separate shapelet sets for each input channel
    - Distance-based feature extraction
    - Preserves initial shapelets for analysis

    Args:
        shapelet_len (int): Length of each shapelet in timesteps
        num_shapelets (int): Number of shapelets to learn per channel

    Shape Notation:
        B: Batch size
        C: Number of channels (fixed at 2)
        T: Time series length
        K: Number of shapelets per channel
        L: Shapelet length

    Example:
        >>> extractor = BatchedShapeletExtractor(shapelet_len=50, num_shapelets=8)
        >>> x = torch.randn(16, 2, 300)  # batch of 16 samples
        >>> features = extractor(x)  # output shape: [16, 16]
    """

    def __init__(self, shapelet_len, num_shapelets):
        super().__init__()
        self.shapelet_len = shapelet_len
        self.num_shapelets = num_shapelets

        # Initialize shapelet parameters
        self.shapelets_ch1 = nn.Parameter(
            torch.randn(num_shapelets, shapelet_len))
        self.shapelets_ch2 = nn.Parameter(
            torch.randn(num_shapelets, shapelet_len))

        # Save initial shapelets for visualization
        self.register_buffer("shapelets_ch1_init",
                             self.shapelets_ch1.detach().clone())
        self.register_buffer("shapelets_ch2_init",
                             self.shapelets_ch2.detach().clone())

    def forward(self, x):
        """
        Compute shapelet-based features via minimum distance matching.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 2, T]

        Returns:
            torch.Tensor: Concatenated features of shape [B, 2*K]

        Process:
            1. Extract subsequences via sliding window
            2. Compute squared distances to all shapelets
            3. Take minimum distance per shapelet
            4. Concatenate features from both channels
        """
        B, C, T = x.shape
        assert C == 2, "Input must have exactly 2 channels"

        ch1, ch2 = x[:, 0, :], x[:, 1, :]
        K, L = self.shapelets_ch1.shape

        def compute_min_dist(signal, shapelets):
            """Helper function to compute minimum distances."""
            dists = []
            for i in range(T - L + 1):
                subseq = signal[:, i:i + L]  # [B, L]
                diff = subseq[:, None, :] - shapelets[None, :, :]  # [B, K, L]
                dist = (diff ** 2).sum(dim=-1)  # [B, K]
                dists.append(dist)
            dists = torch.stack(dists, dim=1)  # [B, num_windows, K]
            return dists.min(dim=1).values  # [B, K]

        dists_ch1 = compute_min_dist(ch1, self.shapelets_ch1)
        dists_ch2 = compute_min_dist(ch2, self.shapelets_ch2)

        return torch.cat([dists_ch1, dists_ch2], dim=1)

    def forward_get_channelwise(self, x):
        """
        Compute average distances per channel for analysis.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 2, T]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Average distances for channel 1 [B, K]
                - Average distances for channel 2 [B, K]
        """
        B, C, T = x.shape
        assert C == 2, "Input must have exactly 2 channels"

        ch1, ch2 = x[:, 0, :], x[:, 1, :]
        K, L = self.shapelets_ch1.shape

        def compute_avg_dist(signal, shapelets):
            """Helper function to compute average distances."""
            dists = []
            for i in range(T - L + 1):
                subseq = signal[:, i:i + L]
                diff = subseq[:, None, :] - shapelets[None, :, :]
                dist = (diff ** 2).sum(dim=-1)
                dists.append(dist)
            dists = torch.stack(dists, dim=1)
            return dists.mean(dim=1)

        return compute_avg_dist(ch1, self.shapelets_ch1), compute_avg_dist(ch2, self.shapelets_ch2)

    def get_raw_shapelets_initial(self):
        """
        Get initial shapelet values (before training).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Initial shapelets for channel 1 and 2, each of shape [K, L]
        """
        return self.shapelets_ch1_init.detach().cpu(), self.shapelets_ch2_init.detach().cpu()

    def get_raw_shapelets_current(self):
        """
        Get current learned shapelet values.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Current shapelets for channel 1 and 2, each of shape [K, L]
        """
        return self.shapelets_ch1.detach().cpu(), self.shapelets_ch2.detach().cpu()


class GNNWithAttention(nn.Module):
    """
    Graph Attention Network with shapelet-based feature extraction.

    Combines:
    - Shapelet learning from time series data
    - Multi-head graph attention layers
    - Graph-level classification

    Architecture:
    1. Shapelet feature extraction
    2. 3-layer GAT with batch normalization
    3. Global mean pooling
    4. MLP classifier

    Args:
        in_channels (int): Unused (for interface compatibility)
        hidden_channels (int): Dimension of hidden layers
        out_channels (int): Number of output classes
        heads (int): Number of attention heads (default: 2)
        shapelet_len (int): Length of shapelets (default: 50)
        num_shapelets (int): Number of shapelets per channel (default: 10)

    Shape Notation:
        N: Number of nodes
        E: Number of edges
        G: Number of graphs
        H: Hidden dimension
    """

    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, shapelet_len=50, num_shapelets=10):
        super().__init__()
        self.shapelet_model = BatchedShapeletExtractor(
            shapelet_len, num_shapelets)

        # Graph attention layers
        self.gat1 = GATConv(num_shapelets * 2, hidden_channels, heads=heads)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.gat2 = GATConv(hidden_channels * heads,
                            hidden_channels, heads=heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels * heads)
        self.gat3 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(p=0.1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels//2, out_channels)
        )

    def forward(self, x, edge_index, batch, mask_channel=None):
        """
        Forward pass for classification.

        Args:
            x (torch.Tensor): Node features [N, 2, T]
            edge_index (torch.LongTensor): Edge indices [2, E]
            batch (torch.LongTensor): Batch indices [N]
            mask_channel (int, optional): Channel to mask (0 or 1)

        Returns:
            torch.Tensor: Class logits [G, out_channels]
        """
        embedding = self.forward_embedding(x, edge_index, batch, mask_channel)
        return self.classifier(embedding)

    def forward_embedding(self, x, edge_index, batch, mask_channel=None):
        """
        Generate graph embeddings through shapelet extraction and GAT layers.

        Args:
            x (torch.Tensor): Node features [N, 2, T]
            edge_index (torch.LongTensor): Edge indices [2, E]
            batch (torch.LongTensor): Batch indices [N]
            mask_channel (int, optional): Channel to mask (0 or 1)

        Returns:
            torch.Tensor: Graph embeddings [G, hidden_channels]
        """
        if mask_channel is not None:
            x = x.clone()
            x[:, mask_channel, :] = 0.0

        # Shapelet feature extraction
        x = self.shapelet_model(x)

        # GAT layers with ELU activation
        x, _ = self.gat1(x, edge_index, return_attention_weights=True)
        x = self.bn1(x)
        x = F.elu(x)

        x, _ = self.gat2(x, edge_index, return_attention_weights=True)
        x = self.bn2(x)
        x = F.elu(x)

        x, _ = self.gat3(x, edge_index, return_attention_weights=True)
        x = self.bn3(x)
        x = F.elu(x)

        # Graph-level pooling
        return global_mean_pool(x, batch)

    def get_shapelets_initial(self):
        """Get initial shapelet values."""
        return self.shapelet_model.get_raw_shapelets_initial()

    def get_shapelets_current(self):
        """Get current learned shapelet values."""
        return self.shapelet_model.get_raw_shapelets_current()
