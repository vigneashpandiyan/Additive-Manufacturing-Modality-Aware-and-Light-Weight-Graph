# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- A baseline 1D Convolutional Neural Network model for multi-channel sensor classification.

Note: Any reuse of this code should be authorized by the code author.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    """
    Classic 1D Convolutional Neural Network processing the dual-channel signals.
    """
    def __init__(self, in_channels=2, num_classes=5):
        """
        Description:
            Initializes 1D CNN network layers including Conv1d layers, BatchNorm layers, and the linear classification layers.
        Purpose:
            To configure structural weights and baseline layers for temporal sequence training.
        Input Types:
            - in_channels (int): Count of input waveforms. Default is 2.
            - num_classes (int): Number of target alloy compositions. Default is 5.
        Output Types:
            - None: Initializes structural layers.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3) # 5000 -> 2500
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)         # 2500 -> 1250
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)        # 1250 -> 625
        self.bn3 = nn.BatchNorm1d(128)
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
            Performs the forward computational pass, feeding input sequence tensors through successive Conv1D blocks, adaptive average pooling, and feed-forward fully connected layer classification.
        Purpose:
            To execute predictive mapping from dual-channel waveforms to predicted class distributions.
        Input Types:
            - x (torch.Tensor): Dual-channel sequence tensor of shape [Batch, Channels, Timesteps].
        Output Types:
            - output (torch.Tensor): Categorical logits array of shape [Batch, NumClasses].
        """
        # x shape: [B, 2, 5000]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1) # [B, 128]
        return self.classifier(x)
