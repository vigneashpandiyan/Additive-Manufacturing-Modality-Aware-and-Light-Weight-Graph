# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- A hybrid Convolutional and Long Short-Term Memory model for temporal feature learning.

Note: Any reuse of this code should be authorized by the code author.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM network that extracts local conv features and models sequential dependencies.
    """
    def __init__(self, in_channels=2, num_classes=5):
        """
        Description:
            Initializes CNNLSTM network layers: Conv1d, BatchNorm, bidirectional LSTM sequence layer, and the fully-connected linear projection head classifier.
        Purpose:
            To define structural states of the hybrid model combining local spatial convolutions and long-range sequential recurrent links.
        Input Types:
            - in_channels (int): Input sensor channel count. Default is 2.
            - num_classes (int): Number of target alloy compositions. Default is 5.
        Output Types:
            - None: Builds structural components.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=4, padding=3)  # 5000 -> 1250
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2)          # 1250 -> 312
        self.bn2 = nn.BatchNorm1d(64)
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), # 64 * 2 = 128 due to bidirectional
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Description:
            Runs the computational pass: Conv1D blocks downsample the waveform, transposes dimensions, extracts recurrent states using bidirectional LSTM, performs global temporal average pooling, and projects output classes.
        Purpose:
            To execute hybrid spatial-recurrent mapping from dual-channel waveforms to alloy classification categories.
        Input Types:
            - x (torch.Tensor): Sequential inputs of shape [Batch, Channels, Timesteps].
        Output Types:
            - logits (torch.Tensor): Output alloy category distribution of shape [Batch, NumClasses].
        """
        # x shape: [B, 2, 5000]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Reshape to sequence: [B, seq_len=312, features=64]
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x) # [B, 312, 128]
        
        # Average pool over time
        out = torch.mean(lstm_out, dim=1) # [B, 128]
        return self.classifier(out)
