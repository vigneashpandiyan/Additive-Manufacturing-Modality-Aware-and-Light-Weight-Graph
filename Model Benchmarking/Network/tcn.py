# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- A Temporal Convolutional Network (TCN) utilizing dilated causal convolutions for sequence modeling.

Note: Any reuse of this code should be authorized by the code author.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """
    A single residual block for Temporal Convolutional Network with causal slicing.
    """
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3):
        """
        Description:
            Initializes layers of a causal dilated residual block (TCNBlock): Conv1D layers with custom dilation and padding, batch normalization, 1x1 downsampling layers, and dropout nodes.
        Purpose:
            To build a single dilation unit capable of modeling temporal causal dependencies in sequence data.
        Input Types:
            - in_channels (int): Incoming feature dimension.
            - out_channels (int): Output projection dimension.
            - dilation (int): Dilation factor coefficient.
            - kernel_size (int): Size of receptive window. Default is 3.
        Output Types:
            - None: Builds structure modules.
        """
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Description:
            Computes causal dilated convolutions by slicing the outputs by `self.padding` elements on the trailing dimension. Normalizes states, applies ReLU non-linearities, performs residual identity skip connections (with optional 1x1 downsampling), and adds them.
        Purpose:
            To perform causal dilated processing on a sequential input.
        Input Types:
            - x (torch.Tensor): Sequential inputs of shape [Batch, Channels, Length].
        Output Types:
            - output (torch.Tensor): Output tensor of shape [Batch, OutChannels, Length].
        """
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
        """
        Description:
            Initializes stacked dilated causal TCNBlocks, adaptive global average pooling, and linear projection classifier head.
        Purpose:
            To construct the complete Temporal Convolutional Network sequence classifier.
        Input Types:
            - in_channels (int): Input channel count. Default is 2.
            - num_classes (int): Number of target alloy compositions. Default is 5.
        Output Types:
            - None: Builds structure modules.
        """
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
        """
        Description:
            Passes input sequences through stacked residual blocks of expanding dilation factors, pools over the temporal dimension, and projects outputs using classifier linear layers.
        Purpose:
            To classify alloy compositions from raw dual-channel waveforms using dilated convolutional receptive fields.
        Input Types:
            - x (torch.Tensor): Sequential inputs of shape [Batch, Channels, Length].
        Output Types:
            - logits (torch.Tensor): Classification logits of shape [Batch, NumClasses].
        """
        # x shape: [B, 2, 5000]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1) # [B, 128]
        return self.classifier(x)
