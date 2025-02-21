import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import orjson
from collections import defaultdict

class SignLanguageDataset(Dataset):
    def __init__(self, file_path, max_frames=233, transform=None):
        print(f"\nLoading dataset from {file_path}...")
        self.transform = transform
        self.data = []
        self.vocab = {}
        self.max_frames = max_frames
        self.expected_frame_length = 49 * 3
        raw_data = self._load_data(file_path)
        print(f"Loaded {len(raw_data)} gloss entries.")
        self._process_data(raw_data)
        print(f"Dataset processing complete. Max sequence length: {self.max_frames}, Vocabulary size: {len(self.vocab)}")

    def _load_data(self, file_path):
        raw_data = defaultdict(list)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ERROR: {file_path} does not exist")
        print("Reading JSONL file line by line...")
        with open(file_path, "rb") as f:
            for line in f:
                item = orjson.loads(line)
                for gloss, videos in item.items():
                    raw_data[gloss].extend(videos)
        return raw_data

    def _process_data(self, raw_data):
        print("Processing dataset and computing max frame count...")
        for gloss, videos in raw_data.items():
            for video in videos:
                self.max_frames = max(self.max_frames, len(video))
        for gloss, videos in raw_data.items():
            if gloss not in self.vocab:
                self.vocab[gloss] = len(self.vocab)
            for video in videos:
                padded_video = self._pad_video(video, self.max_frames)
                self.data.append(torch.tensor(padded_video, dtype=torch.float32))

    def _pad_video(self, video, max_frames):
        processed_frames = []
        for frame in video:
            if len(frame) != 49:
                raise ValueError(f"ERROR: Expected 49 joints per frame, but got {len(frame)}.")
            flat_frame = [coord for joint in frame for coord in joint]
            if len(flat_frame) < self.expected_frame_length:
                flat_frame.extend([0.0] * (self.expected_frame_length - len(flat_frame)))
            processed_frames.append(flat_frame)
        num_frames = len(processed_frames)
        if num_frames < max_frames:
            pad_frame = [0.0] * self.expected_frame_length
            processed_frames.extend([pad_frame] * (max_frames - num_frames))
        else:
            processed_frames = processed_frames[:max_frames]
        return processed_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].reshape(3, self.max_frames, 49)

class LearnableAdjacency(nn.Module):
    def __init__(self, num_nodes):
        super(LearnableAdjacency, self).__init__()
        self.A = nn.Parameter(torch.eye(num_nodes) + 0.01 * torch.randn(num_nodes, num_nodes))
    def forward(self):
        return torch.softmax(self.A, dim=-1)

class STGCN(nn.Module):
    def __init__(self, in_channels=3, num_nodes=49, hidden_dim=128, dropout=0.3):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
dataset = SignLanguageDataset("Dataset/0_landmarks.jsonl")
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
print(f"Dataset split: {len(train_set)} training, {len(val_set)} validation, {len(test_set)} test samples.")
model = STGCN().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
def train_model(train_loader, val_loader, num_epochs=100, patience=10):
    print("\nStarting Training...")
    model.train()
    best_val_loss = float("inf")
    early_stop_counter = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            refined_output = model(batch)
            loss = criterion(refined_output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                val_loss += criterion(output, batch).item()
        val_loss /= len(val_loader)
        model.train()
        print(f"Epoch [{epoch+1}/100] | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "Dataset/stgcn_best_model.pth")
            print(f"Best model saved at epoch {epoch+1}.")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
        scheduler.step()
    print("\nTraining Complete!")


train_model(train_loader, val_loader)
