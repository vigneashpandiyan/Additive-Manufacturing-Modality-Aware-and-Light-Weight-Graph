# -*- coding: utf-8 -*-
"""
GCN without shapelets baseline model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNWithoutShapelets(nn.Module):
    """
    GCN model that processes raw window node features directly by projecting 
    them with a Linear layer, using Graph Convolutional Networks (GCN) instead of GAT.
    """
    def __init__(self, window_size=500, hidden_channels=16, out_channels=5):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(2 * window_size, 20),
            nn.ReLU()
        )
        
        # GCN Layers
        self.gcn1 = GCNConv(20, hidden_channels * 2)
        self.bn1 = nn.BatchNorm1d(hidden_channels * 2)
        self.gcn2 = GCNConv(hidden_channels * 2, hidden_channels * 2)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.gcn3 = GCNConv(hidden_channels * 2, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        self.dropout = nn.Dropout(p=0.1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, edge_index, batch):
        # x shape: [N, 2, window_size]
        N, C, W = x.shape
        x = x.view(N, C * W)
        x = self.projector(x)
        
        # GCN Forward
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        
        x = self.gcn3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        return self.classifier(x)
