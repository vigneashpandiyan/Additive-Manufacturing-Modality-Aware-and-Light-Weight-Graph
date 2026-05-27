# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- BatchedShapeletExtractor: A parallel learnable shapelet matching module mapping waveforms to structural nodes.
- GNNWithAttention: Core Graph Attention Network (GAT) fusing optical/acoustic shapelet metrics via cross-channel attention.

Note: Any reuse of this code should be authorized by the code author.
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
    """
    def __init__(self, shapelet_len, num_shapelets):
        """
        Description:
            Initializes the shapelet matching layer with two sets of randomly initialized learnable parameters (one per sensor channel) and registers buffers storing initial state copies.
        Purpose:
            To configure model capacity and initialize shapelet filters for learning temporal patterns.
        Input Types:
            - shapelet_len (int): Length of each shapelet in timesteps.
            - num_shapelets (int): Number of shapelets to learn per channel.
        Output Types:
            - None: Builds model layers.
        """
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
        Description:
            Splits the dual-channel signals and computes the minimum squared Euclidean distances between the sliding windows of input signals and all learnable shapelets for each channel. Concatenates outputs from both channels.
        Purpose:
            To execute differentiable, distance-based temporal feature extraction from multi-modal waveforms.
        Input Types:
            - x (torch.Tensor): Dual-channel sequence tensor of shape [B, 2, T].
        Output Types:
            - distances (torch.Tensor): Concatenated minimum distance features of shape [B, 2*K].
        """
        B, C, T = x.shape
        assert C == 2, "Input must have exactly 2 channels"

        ch1, ch2 = x[:, 0, :], x[:, 1, :]
        K, L = self.shapelets_ch1.shape

        def compute_min_dist(signal, shapelets):
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
        Description:
            Computes average squared Euclidean distances between sliding windows of the input signals and the learned shapelets.
        Purpose:
            To calculate average channel activation metrics for post-hoc interpretations.
        Input Types:
            - x (torch.Tensor): Dual-channel sequence tensor of shape [B, 2, T].
        Output Types:
            - tuple: (ch1_avg_dist, ch2_avg_dist)
                - ch1_avg_dist (torch.Tensor): Average distances for channel 1 [B, K].
                - ch2_avg_dist (torch.Tensor): Average distances for channel 2 [B, K].
        """
        B, C, T = x.shape
        assert C == 2, "Input must have exactly 2 channels"

        ch1, ch2 = x[:, 0, :], x[:, 1, :]
        K, L = self.shapelets_ch1.shape

        def compute_avg_dist(signal, shapelets):
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
        Description:
            Retrieves the saved copies of shapelet parameters recorded during model initialization.
        Purpose:
            To support comparison of shapelet features before and after optimization.
        Input Types:
            - None
        Output Types:
            - tuple: (ch1_init, ch2_init)
                - ch1_init (torch.Tensor): Initial shapelet weights for channel 1 [K, L].
                - ch2_init (torch.Tensor): Initial shapelet weights for channel 2 [K, L].
        """
        return self.shapelets_ch1_init.detach().cpu(), self.shapelets_ch2_init.detach().cpu()

    def get_raw_shapelets_current(self):
        """
        Description:
            Retrieves the current optimized values of the learnable shapelet parameters.
        Purpose:
            To output current shapelet arrays for analysis and plotting.
        Input Types:
            - None
        Output Types:
            - tuple: (ch1_current, ch2_current)
                - ch1_current (torch.Tensor): Current shapelet weights for channel 1 [K, L].
                - ch2_current (torch.Tensor): Current shapelet weights for channel 2 [K, L].
        """
        return self.shapelets_ch1.detach().cpu(), self.shapelets_ch2.detach().cpu()


class GNNWithAttention(nn.Module):
    """
    Graph Attention Network with shapelet-based feature extraction.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, shapelet_len=50, num_shapelets=10):
        """
        Description:
            Initializes the hybrid GNN model: instantiates the `BatchedShapeletExtractor`, configures three successive GATConv layers with batch normalization, and sets up the final fully-connected MLP classification head.
        Purpose:
            To build the core end-to-end framework integrating local shapelet matching and global graph attention.
        Input Types:
            - in_channels (int): Unused parameter kept for interface compatibility.
            - hidden_channels (int): Hidden dimensions size.
            - out_channels (int): Number of target alloy compositions.
            - heads (int): Number of attention heads for GAT blocks. Default is 2.
            - shapelet_len (int): Length of each shapelet filter. Default is 50.
            - num_shapelets (int): Shapelets per modality. Default is 10.
        Output Types:
            - None: Builds structure modules.
        """
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
        Description:
            Runs the full classification pipeline: extracts GAT embeddings from the input node sequences and maps them to alloy composition class logits.
        Purpose:
            To execute forward mapping from raw window node sequences to output predictions.
        Input Types:
            - x (torch.Tensor): Window nodes sequence tensor of shape [N, 2, T].
            - edge_index (torch.Tensor): Connectivity adjacency matrix of shape [2, E].
            - batch (torch.Tensor): Graph assignment index map of shape [N].
            - mask_channel (int, optional): Modality channel index to mask out (0 or 1) for ablation testing.
        Output Types:
            - logits (torch.Tensor): Categorical predictions of shape [B, OutChannels].
        """
        embedding = self.forward_embedding(x, edge_index, batch, mask_channel)
        return self.classifier(embedding)

    def forward_embedding(self, x, edge_index, batch, mask_channel=None):
        """
        Description:
            Generates graph representations by extracting shapelet features (optionally masking one modality), passing them through stacked multi-head GATConv layers with ELU activations and batch normalizations, and performing global mean pooling across node states.
        Purpose:
            To map node sequences to unified, explainable graph embeddings.
        Input Types:
            - x (torch.Tensor): Window nodes sequence tensor of shape [N, 2, T].
            - edge_index (torch.Tensor): Connectivity adjacency matrix of shape [2, E].
            - batch (torch.Tensor): Graph assignment index map of shape [N].
            - mask_channel (int, optional): Modality channel index to mask out (0 or 1) for ablation testing.
        Output Types:
            - graph_embeddings (torch.Tensor): Fused graph representations of shape [B, HiddenChannels].
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
        """
        Description:
            Retrieves initial shapelets parameters.
        Purpose:
            Helper mapping to support shapelet analysis.
        Input Types:
            - None
        Output Types:
            - tuple: Initial shapelets for channel 1 and 2.
        """
        return self.shapelet_model.get_raw_shapelets_initial()

    def get_shapelets_current(self):
        """
        Description:
            Retrieves optimized shapelets parameters.
        Purpose:
            Helper mapping to support shapelet analysis.
        Input Types:
            - None
        Output Types:
            - tuple: Current shapelets for channel 1 and 2.
        """
        return self.shapelet_model.get_raw_shapelets_current()
