import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error


class ProbabilisticCNN(nn.Module):
    def __init__(self, num_features=324, num_frames=300): # Note: 324 Features now
        super().__init__()

        # Attention Layer
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_features // 4),
            nn.ReLU(),
            nn.Linear(num_features // 4, num_features),
            nn.Sigmoid()
        )

        # 1D CNN
        self.conv_layers = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Dual Head Regressor (Mean + Uncertainty)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2) # [Mean, LogVar]
        )

    def forward(self, x):
        # x input: (Batch, 300, 324)
        x = x.permute(0, 2, 1) # (Batch, 324, 300)

        # Apply Attention
        # The model should learn to give High Weight to Channel 324!
        attn_weights = self.attention(x).unsqueeze(2)
        x = x * attn_weights

        x = self.conv_layers(x)
        out = self.regressor(x)

        mean = torch.nn.functional.softplus(out[:, 0]) # Positive count
        log_var = out[:, 1]

        return mean, log_var