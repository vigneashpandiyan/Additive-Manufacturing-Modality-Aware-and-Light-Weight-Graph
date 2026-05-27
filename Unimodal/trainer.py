# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- `BatchedShapeletExtractor`: A parallel learnable shapelet matching module mapping waveforms to structural nodes.
- `GNNWithAttention`: Graph neural network with GATConv and classification MLP adapted for unimodal input data.

Note: Any reuse of this code should be authorized by the code author.
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
        """
        Description:
            Initializes the BatchedShapeletExtractor module, instantiating learnable shapelets
            and registering a buffer to keep a copy of their initial state.
        Purpose:
            To configure shapelet lengths and counts, and preserve an initial baseline reference.
        Input Types:
            - shapelet_len (int): Length of each shapelet.
            - num_shapelets (int): Number of shapelets to learn.
        Output Types:
            - None
        """
        super().__init__()
        self.shapelet_len = shapelet_len
        self.num_shapelets = num_shapelets

        # Learnable shapelets for 1 channel
        self.shapelets = nn.Parameter(torch.randn(num_shapelets, shapelet_len))

        # Save initial shapelets
        self.register_buffer("shapelets_init", self.shapelets.detach().clone())

    def forward(self, x):
        """
        Description:
            Computes the minimum distance-based shapelet feature values over the input time series.
        Purpose:
            To extract shapelet matching distance signatures from multi-window time series.
        Input Types:
            - x (torch.Tensor): Input tensor of shape (B, 1, T) representing batch size, channel count, and temporal length.
        Output Types:
            - min_dists (torch.Tensor): Computed distance features of shape (B, K), where K is the number of shapelets.
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
        Description:
            Computes average distance scores between shapelets and all sliding windows.
        Purpose:
            To provide detailed sequence activation/matching scores for explainability and feature visualization.
        Input Types:
            - x (torch.Tensor): Input signal tensor of shape (B, 1, T).
        Output Types:
            - avg_dists (torch.Tensor): Averaged shapelet distance features of shape (B, K).
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
        Description:
            Retrieves a copy of the initial, untrained shapelet weights stored in the registered buffer.
        Purpose:
            To provide a comparative baseline to visualize how shapelets evolve during model training.
        Input Types:
            - None
        Output Types:
            - initial_shapelets (torch.Tensor): Initial shapelet weights of shape (K, L).
        """
        return self.shapelets_init.detach().cpu()

    def get_raw_shapelets_current(self):
        """
        Description:
            Retrieves the current, trained shapelet weights.
        Purpose:
            To visualize the final learned shapelets representing key temporal signatures.
        Input Types:
            - None
        Output Types:
            - current_shapelets (torch.Tensor): Current shapelet weights of shape (K, L).
        """
        return self.shapelets.detach().cpu()

    def forward_get_channelwise(self, x):
        """
        Description:
            Computes average shapelet distances separately for the first channel of the input.
        Purpose:
            To provide single-channel shapelet match profiles from multi-channel input structures.
        Input Types:
            - x (torch.Tensor): Input tensor of shape (B, 2, T).
        Output Types:
            - dist_ch1 (torch.Tensor): Channel-wise averaged shapelet distances of shape (B, K).
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
        """
        Description:
            Initializes the GNNWithAttention model with a BatchedShapeletExtractor, 
            three GATConv graph attention layers, BatchNorm layers, and a classification MLP.
        Purpose:
            To build the full single-channel graph neural network architecture using dynamic shapelet extraction.
        Input Types:
            - in_channels (int): Interface parameter, not directly used in extraction.
            - hidden_channels (int): Number of hidden channels in GAT Conv layers.
            - out_channels (int): Number of target classification classes.
            - heads (int): Number of attention heads for multi-head GAT (default: 2).
            - shapelet_len (int): Subsequence time-series length of learned shapelets (default: 50).
            - num_shapelets (int): Number of shapelets learned per channel (default: 10).
        Output Types:
            - None
        """
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
        Description:
            Runs the full forward inference pass of the unimodal GNN model on input graphs.
        Purpose:
            To map a batch of graphs to class probability logits.
        Input Types:
            - x (torch.Tensor): Node features tensor of shape (num_nodes, 1, T).
            - edge_index (torch.LongTensor): Graph edge connectivity index of shape (2, E).
            - batch (torch.LongTensor): Mapping vector of shape (num_nodes,) assigning nodes to graphs.
        Output Types:
            - logits (torch.Tensor): Classification logits of shape (num_graphs, out_channels).
        """
        embedding = self.forward_embedding(x, edge_index, batch)
        return self.classifier(embedding)

    def forward_embedding(self, x, edge_index, batch):
        """
        Description:
            Runs a forward pass through the shapelet extractor and GAT attention conv blocks
            to compute global pooled graph embeddings.
        Purpose:
            To extract high-level representative graph embeddings for downstream classification.
        Input Types:
            - x (torch.Tensor): Raw temporal node feature tensor of shape (num_nodes, 1, T).
            - edge_index (torch.LongTensor): Graph edge connections of shape (2, E).
            - batch (torch.LongTensor): Node-to-graph grouping vector.
        Output Types:
            - embeddings (torch.Tensor): Global graph representation tensor of shape (num_graphs, hidden_channels).
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
        Description:
            Retrieves the initial (untrained) shapelet weights from the extractor sub-module.
        Purpose:
            To provide the initial reference shapelet weights for comparative visualization.
        Input Types:
            - None
        Output Types:
            - initial_shapelets (torch.Tensor): Initial shapelet weights of shape (K, L).
        """
        return self.shapelet_model.get_raw_shapelets_initial()

    def get_shapelets_current(self):
        """
        Description:
            Retrieves the currently learned shapelet weights from the extractor sub-module.
        Purpose:
            To provide the trained shapelet weights for visual inspections.
        Input Types:
            - None
        Output Types:
            - current_shapelets (torch.Tensor): Current shapelet weights of shape (K, L).
        """
        return self.shapelet_model.get_raw_shapelets_current()
