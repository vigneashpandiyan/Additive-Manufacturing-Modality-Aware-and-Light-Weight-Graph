# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- A baseline multi-head self-attention Transformer model adapted for sensor sequence classification.

Note: Any reuse of this code should be authorized by the code author.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerClassifier(nn.Module):
    """
    Transformer Encoder Classifier for time series classification.
    """
    def __init__(self, in_channels=2, num_classes=5, d_model=64, nhead=4, num_layers=2):
        """
        Description:
            Initializes layers of the TransformerClassifier: downsampling Conv1D projection, position embeddings, TransformerEncoder layers with nheads, and the linear classification head.
        Purpose:
            To build standard self-attention based model architecture baseline.
        Input Types:
            - in_channels (int): Input channel count. Default is 2.
            - num_classes (int): Number of target alloy compositions. Default is 5.
            - d_model (int): Inner hidden state projection size. Default is 64.
            - nhead (int): Multi-head attention head count. Default is 4.
            - num_layers (int): Count of stacked transformer blocks. Default is 2.
        Output Types:
            - None: Builds structure modules.
        """
        super().__init__()
        # Downsample sequence to length 100 via a Conv1d to prevent memory/computation blowup
        self.projection = nn.Conv1d(in_channels, d_model, kernel_size=50, stride=50) # 5000 -> 100
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        Description:
            Projects multi-channel sequence vectors, adds positional parameter offsets, runs multi-head self-attention transformer blocks, pools values via temporal mean aggregation, and scores categories using linear classifier layers.
        Purpose:
            To execute self-attention classification from raw sensory input sequences.
        Input Types:
            - x (torch.Tensor): Sequential inputs of shape [Batch, Channels, Timesteps].
        Output Types:
            - logits (torch.Tensor): Classification logits of shape [Batch, NumClasses].
        """
        # x shape: [B, 2, 5000]
        x = self.projection(x)      # [B, d_model, 100]
        x = x.transpose(1, 2)       # [B, 100, d_model]
        x = x + self.pos_encoder
        x = self.transformer_encoder(x) # [B, 100, d_model]
        
        out = torch.mean(x, dim=1)  # Pool over sequence dimension: [B, d_model]
        return self.classifier(out)
