# -*- coding: utf-8 -*-
"""
Temporal Convolutional Network (TCN) baseline model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """
    A single residual block for Temporal Convolutional Network with causal slicing.
    """
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]  # Causal slice
        out = F.relu(self.bn1(out))
        out = self.dropout(out)
        
        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]  # Causal slice
        out = F.relu(self.bn2(out))
        out = self.dropout(out)
        
        if self.downsample is not None:
            res = self.downsample(res)
            
        return F.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network consisting of stacked dilated residual blocks.
    """
    def __init__(self, in_channels=2, num_classes=5):
        super().__init__()
        self.block1 = TCNBlock(in_channels, 32, dilation=1)
        self.block2 = TCNBlock(32, 64, dilation=2)
        self.block3 = TCNBlock(64, 128, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [B, 2, 5000]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1) # [B, 128]
        return self.classifier(x)
