# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 12:01:31 2025

@author: vpsora

Any reuse of this code should be authorized by the code author.
Developed for the publication:
"Modality-Aware and Light-Weight Graph Attention Networkfor In-SituComposition Monitoring 
in PBF-LB of Graded 316L–CuCrZr Alloys by Sensor Fusion of Optical and Acoustic Emissions"

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class BatchedShapeletExtractor(nn.Module):

    """
    Learns shapelets for a single-channel time series and extracts features by computing
    the minimum (or average) distance between each shapelet and sliding windows of the input.

    Attributes:
    -----------
    shapelet_len : int
        Length of each shapelet (L).
    num_shapelets : int
        Number of shapelets (K).
    shapelets : nn.Parameter
        Learnable shapelet tensor of shape (K, L).
    """

    def __init__(self, shapelet_len, num_shapelets):
        super().__init__()
        self.shapelet_len = shapelet_len
        self.num_shapelets = num_shapelets

        # Learnable shapelets for 1 channel
        self.shapelets = nn.Parameter(torch.randn(num_shapelets, shapelet_len))

        # Save initial shapelets
        self.register_buffer("shapelets_init", self.shapelets.detach().clone())

    def forward(self, x):
        """
        Computes minimum distance-based shapelet features.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (B, 1, T), where:
            - B is batch size
            - 1 is the channel count
            - T is time series length

        Returns:
        --------
        torch.Tensor
            Distance features of shape (B, K), where K is num_shapelets.
        """
        B, C, T = x.shape
        assert C == 1, "Expected single-channel input (C=1)"
        # print(f"[Forward] Input shape: {x.shape}")  # [B, 1, T]

        signal = x[:, 0, :]  # [B, T]
        K, L = self.shapelets.shape

        dists = []
        for i in range(T - L + 1):
            subseq = signal[:, i:i + L]  # [B, L]
            diff = subseq[:, None, :] - self.shapelets[None, :, :]  # [B, K, L]
            dist = (diff ** 2).sum(dim=-1)  # [B, K]
            dists.append(dist)

        dists = torch.stack(dists, dim=1)  # [B, num_windows, K]
        min_dists = dists.min(dim=1).values  # [B, K]
        # print(f"[Forward] Output shape: {min_dists.shape}")  # [B, K]
        return min_dists

    def forward_get_detailed(self, x):
        """
        Computes average distances between shapelets and each window.
        Useful for interpretability or saliency.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (B, 1, T)

        Returns:
        --------
        torch.Tensor
            Average distances for each shapelet: shape (B, K)
        """
        B, C, T = x.shape
        assert C == 1, "Expected single-channel input (C=1)"
        # print(f"[Detailed] Input shape: {x.shape}")  # [B, 1, T]

        signal = x[:, 0, :]  # [B, T]
        K, L = self.shapelets.shape

        dists = []
        for i in range(T - L + 1):
            subseq = signal[:, i:i + L]  # [B, L]
            diff = subseq[:, None, :] - self.shapelets[None, :, :]  # [B, K, L]
            dist = (diff ** 2).sum(dim=-1)  # [B, K]
            dists.append(dist)

        dists = torch.stack(dists, dim=1)  # [B, num_windows, K]
        avg_dists = dists.mean(dim=1)  # [B, K]
        # print(f"[Detailed] Output shape: {avg_dists.shape}")
        return avg_dists

    def get_raw_shapelets_initial(self):
        """
        Returns the initial (untrained) shapelets.

        Returns:
        --------
        torch.Tensor of shape (K, L)
        """
        return self.shapelets_init.detach().cpu()

    def get_raw_shapelets_current(self):
        """
        Returns the current (trained) shapelets.

        Returns:
        --------
        torch.Tensor of shape (K, L)
        """
        return self.shapelets.detach().cpu()

    def forward_get_channelwise(self, x):
        """
        Compute average shapelet distances separately for each channel.

        Args:
            x: Tensor of shape [B, 2, T]

        Returns:
            Tuple: (dist_ch1, dist_ch2), each of shape [B, num_shapelets]
        """
        B, C, T = x.shape

        ch1 = x[:, 0, :]
        K, L = self.shapelets_ch1.shape

        def compute_avg_dist(signal, shapelets):
            dists = []
            for i in range(T - L + 1):
                subseq = signal[:, i:i + L]                    # [B, L]
                diff = subseq[:, None, :] - shapelets[None, :, :]  # [B, K, L]
                dist = (diff ** 2).sum(dim=-1)                 # [B, K]
                dists.append(dist)
            # [B, num_positions, K]
            dists = torch.stack(dists, dim=1)
            return dists.mean(dim=1)                           # [B, K]

        dist_ch1 = compute_avg_dist(ch1, self.shapelets_ch1)   # [B, K]

        return dist_ch1


class GNNWithAttention(nn.Module):

    """
    Graph neural network with shapelet-based input encoding for mono-channel time series graphs.

    Modules:
    --------
    - BatchedShapeletExtractorSingleChannel: Learns shapelets for a single input channel.
    - GATConv layers: Three-layer Graph Attention Network.
    - BatchNorm and Dropout: Improve generalization.
    - Classifier: Fully connected MLP for graph-level classification.

    Parameters:
    -----------
    in_channels : int
        Not used directly; kept for interface consistency.
    hidden_channels : int
        Hidden units in GAT layers.
    out_channels : int
        Number of output classes.
    heads : int
        Attention heads per GAT layer.
    shapelet_len : int
        Length of each shapelet.
    num_shapelets : int
        Number of shapelets per sample.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, shapelet_len=50, num_shapelets=10):
        super().__init__()
        self.shapelet_model = BatchedShapeletExtractor(
            shapelet_len, num_shapelets)

        # Input shapelet output: [num_nodes, K]
        self.gat1 = GATConv(num_shapelets, hidden_channels, heads=heads)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.gat2 = GATConv(hidden_channels * heads,
                            hidden_channels, heads=heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels * heads)
        self.gat3 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(p=0.1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels//2, out_channels)
        )

    def forward(self, x, edge_index, batch):
        """
        Forward pass for classification.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (num_nodes, 1, T)
        edge_index : torch.LongTensor
            Edge list (2, E)
        batch : torch.LongTensor
            Batch vector mapping nodes to graphs

        Returns:
        --------
        torch.Tensor
            Class logits of shape (num_graphs, out_channels)
        """
        embedding = self.forward_embedding(x, edge_index, batch)
        return self.classifier(embedding)

    def forward_embedding(self, x, edge_index, batch):
        """
        Forward pass through shapelet extractor and GNN.

        Returns:
        --------
        torch.Tensor
            Graph-level embedding after GNN and pooling.
        """
        # print(f"[Input] x shape: {x.shape}")  # [num_nodes, 1, T]

        x = self.shapelet_model(x)  # [num_nodes, K]
        # print(f"[Shapelet Features] x shape: {x.shape}")

        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        # print(f"[GAT1] x shape: {x.shape}")

        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        # print(f"[GAT2] x shape: {x.shape}")

        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        # print(f"[GAT3] x shape: {x.shape}")

        x = global_mean_pool(x, batch)
        # print(f"[Global Pool] x shape: {x.shape}")

        x = self.dropout(x)
        return x

    def get_shapelets_initial(self):
        """
        Returns:
        --------
        torch.Tensor
            Initial (untrained) shapelets of shape (K, L)
        """
        return self.shapelet_model.get_raw_shapelets_initial()

    def get_shapelets_current(self):
        """
        Returns:
        --------
        torch.Tensor
            Current (learned) shapelets of shape (K, L)
        """
        return self.shapelet_model.get_raw_shapelets_current()
