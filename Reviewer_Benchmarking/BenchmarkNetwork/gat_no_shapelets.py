# -*- coding: utf-8 -*-
"""
GAT without shapelets baseline model architecture.
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
