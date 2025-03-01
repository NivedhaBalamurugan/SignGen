import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch.optim.lr_scheduler import StepLR
import numpy as np
from config import *
from utils.data_utils import load_skeleton_sequences
import cvae_inference, cgan_inference
from architectures.stgcn import STGCN, create_edge_index, group_joints



def temporal_consistency_loss(input_sequence, refined_sequence):
    return F.mse_loss(input_sequence, refined_sequence)

def prepare_data():
    inputs = []
    outputs = []
    skeleton_data = load_skeleton_sequences([FINAL_JSONL_PATHS])
    
    for gloss, videos in skeleton_data.items():
        num_videos = len(videos)
    
        for _ in range(num_videos):
            generated_cvae = cvae_inference.get_cvae_sequence(gloss, 0)
            generated_cgan = cgan_inference.get_cgan_sequence(gloss, 0)
            outputs.append(group_joints(generated_cvae))  # Group generated sequences
            outputs.append(group_joints(generated_cgan))  # Group generated sequences
    
        inputs.extend([group_joints(video) for video in videos])  # Group input sequences
        inputs.extend([group_joints(video) for video in videos])  # Group input sequences
    
    inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
    outputs = torch.tensor(np.array(outputs), dtype=torch.float32)
    return inputs, outputs

def train_stgcn(model, inputs, outputs, edge_index, num_epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    best_loss = float('inf')
    patience = 10
    counter = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(inputs, edge_index)
        recon_loss = F.mse_loss(predictions, outputs)
        temporal_loss = temporal_consistency_loss(inputs, predictions)
        total_loss = recon_loss + temporal_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            counter = 0
            model_save_path = os.path.join(STGCN_MODEL_PATH, "stgcn.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f} (Best model saved)")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}")

    print("Training completed. Best model saved.")

def main():
    print("Preparing data...")
    inputs, outputs = prepare_data()
    edge_index = create_edge_index()
    print("Data preparation complete.")

    print("Initializing ST-GCN model...")
    model = STGCN(num_nodes=16, num_features=3, num_classes=3)
    print("Model initialized.")

    print("Starting model training...")
    train_stgcn(model, inputs, outputs, edge_index)
    print("Model training complete.")

if __name__ == "__main__":
    main()