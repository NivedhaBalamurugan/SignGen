import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch.optim.lr_scheduler import StepLR
import numpy as np
from config import *


class STGCN(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes, hidden_dim=64):
        super(STGCN, self).__init__()
        self.gcn1 = GCNConv(num_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.tcn = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = x.permute(0, 2, 1)
        x = F.relu(self.tcn(x))
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x

def create_edge_index():
    edge_index = [
        [0, 1], [1, 0], [1, 2], [2, 1], [0, 3], [3, 0],
        [1, 4], [4, 1], [4, 5], [5, 4], [4, 6], [6, 4],
        [4, 7], [7, 4], [4, 8], [8, 4], [4, 9], [9, 4],
        [1, 10], [10, 1], [10, 11], [11, 10], [10, 12], [12, 10],
        [10, 13], [13, 10], [10, 14], [14, 10], [10, 15], [15, 10],
    ]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def group_joints(skeleton_sequence):
    num_frames = skeleton_sequence.shape[0]
    grouped_sequence = np.zeros((num_frames, 16, 3))
    grouped_sequence[:, 0, :] = skeleton_sequence[:, 6, :]
    grouped_sequence[:, 1, :] = skeleton_sequence[:, [0, 2], :].mean(axis=1)
    grouped_sequence[:, 2, :] = skeleton_sequence[:, [1, 3], :].mean(axis=1)
    grouped_sequence[:, 3, :] = skeleton_sequence[:, [4, 5], :].mean(axis=1)
    grouped_sequence[:, 4, :] = skeleton_sequence[:, 7, :]
    grouped_sequence[:, 5, :] = skeleton_sequence[:, 8:12, :].mean(axis=1)
    grouped_sequence[:, 6, :] = skeleton_sequence[:, 12:16, :].mean(axis=1)
    grouped_sequence[:, 7, :] = skeleton_sequence[:, 16:20, :].mean(axis=1)
    grouped_sequence[:, 8, :] = skeleton_sequence[:, 20:24, :].mean(axis=1)
    grouped_sequence[:, 9, :] = skeleton_sequence[:, 24:28, :].mean(axis=1)
    grouped_sequence[:, 10, :] = skeleton_sequence[:, 28, :]
    grouped_sequence[:, 11, :] = skeleton_sequence[:, 29:33, :].mean(axis=1)
    grouped_sequence[:, 12, :] = skeleton_sequence[:, 33:37, :].mean(axis=1)
    grouped_sequence[:, 13, :] = skeleton_sequence[:, 37:41, :].mean(axis=1)
    grouped_sequence[:, 14, :] = skeleton_sequence[:, 41:45, :].mean(axis=1)
    grouped_sequence[:, 15, :] = skeleton_sequence[:, 45:49, :].mean(axis=1)
    return grouped_sequence