# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- A Graph Attention Network (GAT) baseline utilizing raw or statistical features directly, bypassing shapelet learning.

Note: Any reuse of this code should be authorized by the code author.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATWithoutShapelets(nn.Module):
    """
    GAT model that processes the raw window node features directly by projecting 
    them with a Linear layer, bypassing the Shapelet Extractor.
    """
    def __init__(self, window_size=500, hidden_channels=16, out_channels=5, heads=2):
        """
        Description:
            Initializes the layers of the baseline GAT model (GATWithoutShapelets) including the linear feature projection layer, sequential GAT attention steps, layer batch normalizations, and the feed-forward output classifier.
        Purpose:
            To build model components to evaluate graph attention performance on flattened raw waveform sequences without extracting shapelets.
        Input Types:
            - window_size (int): Size of the temporal window segment. Default is 500.
            - hidden_channels (int): Dimensional depth of hidden layers. Default is 16.
            - out_channels (int): Output category size. Default is 5.
            - heads (int): Number of attention heads for GAT operations. Default is 2.
        Output Types:
            - None: Builds structure modules.
        """
        super().__init__()
        # Input size: 2 channels * window_size (500) = 1000
        # Project to 20 dimensions (same feature size as 2 * num_shapelets = 20)
        self.projector = nn.Sequential(
            nn.Linear(2 * window_size, 20),
            nn.ReLU()
        )
        
        # GAT Layers (identical to original GNNWithAttention)
        self.gat1 = GATConv(20, hidden_channels, heads=heads)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels * heads)
        self.gat3 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        self.dropout = nn.Dropout(p=0.1)
        
        # Classifier (identical structure to original)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, edge_index, batch):
        """
        Description:
            Flashes the temporal input vector, runs linear projection to size 20, applies successive GATConv layers with ELU non-linearities and batch-normalizations, pools node states via global mean aggregation, and maps outputs using the classifier MLPs.
        Purpose:
            To execute forward computation for a graph model that uses attention without explicit shapelet mapping.
        Input Types:
            - x (torch.Tensor): Window nodes sequence tensor of shape [N, 2, WindowSize].
            - edge_index (torch.Tensor): Node connectivity adjacency matrix of shape [2, E].
            - batch (torch.Tensor): Graph assignment index map of shape [N].
        Output Types:
            - logits (torch.Tensor): Classification predictions of shape [B, OutChannels].
        """
        # x shape: [N, 2, window_size]
        N, C, W = x.shape
        x = x.view(N, C * W)  # Flatten to [N, 1000]
        x = self.projector(x) # Project to [N, 20]
        
        # GAT Forward
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        
        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        return self.classifier(x)
