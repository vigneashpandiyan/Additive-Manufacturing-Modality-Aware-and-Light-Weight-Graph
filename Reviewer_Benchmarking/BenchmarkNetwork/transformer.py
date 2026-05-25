# -*- coding: utf-8 -*-
"""
Transformer Encoder baseline model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerClassifier(nn.Module):
    """
    Transformer Encoder Classifier for time series classification.
    """
    def __init__(self, in_channels=2, num_classes=5, d_model=64, nhead=4, num_layers=2):
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
        # x shape: [B, 2, 5000]
        x = self.projection(x)      # [B, d_model, 100]
        x = x.transpose(1, 2)       # [B, 100, d_model]
        x = x + self.pos_encoder
        x = self.transformer_encoder(x) # [B, 100, d_model]
        
        out = torch.mean(x, dim=1)  # Pool over sequence dimension: [B, d_model]
        return self.classifier(out)
