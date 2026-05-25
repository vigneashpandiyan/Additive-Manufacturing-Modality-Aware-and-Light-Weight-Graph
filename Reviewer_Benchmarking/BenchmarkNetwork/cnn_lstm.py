# -*- coding: utf-8 -*-
"""
CNN-LSTM hybrid baseline model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM network that extracts local conv features and models sequential dependencies.
    """
    def __init__(self, in_channels=2, num_classes=5):
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
        # x shape: [B, 2, 5000]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Reshape to sequence: [B, seq_len=312, features=64]
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x) # [B, 312, 128]
        
        # Average pool over time
        out = torch.mean(lstm_out, dim=1) # [B, 128]
        return self.classifier(out)
