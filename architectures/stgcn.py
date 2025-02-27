import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from config import *


class LearnableAdjacency(nn.Module):
    def __init__(self, num_nodes):
        super(LearnableAdjacency, self).__init__()
        self.A = nn.Parameter(torch.eye(num_nodes) + 0.01 * torch.randn(num_nodes, num_nodes))
    def forward(self):
        return torch.softmax(self.A, dim=-1)

class STGCN(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, num_nodes=NUM_NODES, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super(STGCN, self).__init__()
        print("Initializing ST-GCN model...")
        self.A = LearnableAdjacency(num_nodes)
        self.spatial_conv1 = nn.Linear(in_channels, hidden_dim)
        self.spatial_conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0))
        self.temporal_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0))
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.refine = nn.Linear(hidden_dim, in_channels)
        print("ST-GCN model initialized successfully!")

    def forward(self, x):
        A = self.A()
        B, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3)
        x = torch.matmul(x, A)
        x = x.permute(0, 1, 3, 2)
        x = self.spatial_conv1(x)
        x = F.relu(x)
        x = x.reshape(B * T * V, -1)
        x = self.batch_norm1(x)
        x = x.reshape(B, T, V, -1)
        x = self.dropout(x)
        x = self.spatial_conv2(x)
        x = F.relu(x)
        x = x.reshape(B * T * V, -1)
        x = self.batch_norm2(x)
        x = x.reshape(B, T, V, -1)
        x = self.dropout(x)
        x = x.permute(0, 3, 1, 2)
        x = self.temporal_conv1(x)
        x = self.temporal_conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.refine(x)
        x = x.permute(0, 3, 1, 2)
        return x
