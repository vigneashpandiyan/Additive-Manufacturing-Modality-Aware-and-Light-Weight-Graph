# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- A Graph Convolutional Network (GCN) baseline utilizing structural adjacency without shapelet extractions.

Note: Any reuse of this code should be authorized by the code author.
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
        """
        Description:
            Initializes structural layers of the baseline GCN (GCNWithoutShapelets) model including sequential GCNConv blocks, layer normalization, and classification modules.
        Purpose:
            To build a standard graph convolution architecture baseline that lacks multi-head attention and shapelet matches.
        Input Types:
            - window_size (int): Size of the temporal window segment. Default is 500.
            - hidden_channels (int): Hidden dimensions size. Default is 16.
            - out_channels (int): Output category count. Default is 5.
        Output Types:
            - None: Builds structural components.
        """
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
        """
        Description:
            Flashes and flattens node segments, projects them, runs successive graph convolutional layers with batch normalizations and ELU activations, pools via global average operator, and passes output through classification linear layers.
        Purpose:
            To execute forward computation for a baseline GCN model using structural adjacency.
        Input Types:
            - x (torch.Tensor): Window nodes sequence tensor of shape [N, 2, WindowSize].
            - edge_index (torch.Tensor): Connectivity adjacency matrix of shape [2, E].
            - batch (torch.Tensor): Graph assignment index map of shape [N].
        Output Types:
            - logits (torch.Tensor): Classification logits of shape [B, OutChannels].
        """
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
