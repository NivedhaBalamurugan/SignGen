import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import show_output

class LearnableAdjacency(nn.Module):
    def __init__(self, num_nodes):
        super(LearnableAdjacency, self).__init__()
        self.A = nn.Parameter(torch.eye(num_nodes) + 0.01 * torch.randn(num_nodes, num_nodes))
    def forward(self):
        return torch.softmax(self.A, dim=-1)

class STGCN(nn.Module):
    def __init__(self, in_channels=3, num_nodes=49, hidden_dim=128, dropout=0.3):
        super(STGCN, self).__init__()
        self.A = LearnableAdjacency(num_nodes)
        self.spatial_conv1 = nn.Linear(in_channels, hidden_dim)
        self.spatial_conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0))
        self.temporal_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0))
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.refine = nn.Linear(hidden_dim, in_channels)
    def forward(self, x):
        # x shape: (B, C, T, V)
        A = self.A()
        B, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3)       # (B, T, C, V)
        x = torch.matmul(x, A)          # (B, T, C, V)
        x = x.permute(0, 1, 3, 2)       # (B, T, V, C)
        x = self.spatial_conv1(x)       # (B, T, V, hidden_dim)
        x = F.relu(x)
        x = x.reshape(B * T * V, -1)
        x = self.batch_norm1(x)
        x = x.reshape(B, T, V, -1)
        x = self.dropout(x)
        x = self.spatial_conv2(x)       # (B, T, V, hidden_dim)
        x = F.relu(x)
        x = x.reshape(B * T * V, -1)
        x = self.batch_norm2(x)
        x = x.reshape(B, T, V, -1)
        x = self.dropout(x)
        x = x.permute(0, 3, 1, 2)       # (B, hidden_dim, T, V)
        x = self.temporal_conv1(x)      # (B, hidden_dim, T, V)
        x = self.temporal_conv2(x)      # (B, hidden_dim, T, V)
        x = x.permute(0, 2, 3, 1)       # (B, T, V, hidden_dim)
        x = self.refine(x)              # (B, T, V, in_channels)
        x = x.permute(0, 3, 1, 2)       # (B, in_channels, T, V)
        return x

def load_model(model_path, device):
    model = STGCN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def refine_sequence(model, sequence, device):
    with torch.no_grad():
        if sequence.dim() == 3:
            sequence = sequence.unsqueeze(0)
        sequence = sequence.to(device)
        refined = model(sequence)
    return refined.squeeze(0)

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_path = "Dataset/stgcn_best.pth"
#     if not os.path.exists(model_path):
#         print("Model file not found. Please ensure the best model is saved at", model_path)
#         exit(1)
#     model = load_model(model_path, device)
#     sample_sequence = torch.rand(3, 233, 49)
#     refined_sequence = refine_sequence(model, sample_sequence, device)
#     print("Refined sequence shape:", refined_sequence.shape)


def refine_sequence(input_sequence,model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "Dataset/stgcn_best_model.pth"
    if not os.path.exists(model_path):
        print("Model file not found. Please ensure the best model is saved at", model_path)
        exit(1)
    model = load_model(model_path, device)
    refined_sequence = refine_sequence(model, input_sequence, device)
    print("Refined sequence shape:", refined_sequence.shape)
    show_output.save_generated_sequence(refined_sequence, "Dataset/refined_output_sequence_{model_name}") 
    return refined_sequence

    