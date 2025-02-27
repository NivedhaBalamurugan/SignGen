import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import glob
import os
import orjson
import warnings
from config import *
from architectures.stgcn import LearnableAdjacency, STGCN

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SignLanguageDataset(Dataset):
    def __init__(self, file_path, transform=None):
        logging.info(f"Loading dataset from {file_path}...")
        self.transform = transform
        self.data = []
        self._load_process_data(file_path)
        logging.info(f"Loaded {len(self.data)} gloss entries.")
        logging.info(f"Dataset processing complete. Max sequence length: {MAX_FRAMES}")

    def _load_process_data(self, file_paths):        
        
        file_paths = glob.glob(file_paths)
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                for line in f:
                    try:
                        item = orjson.loads(line)
                        for gloss, videos in item.items():
                            for video in videos:
                                padded_video = self._pad_video(video)
                                
                                try:
                                    padded_video = np.array(padded_video, dtype=np.float32)
                                except Exception as e:
                                    logging.error(f"Error converting video to numpy array for gloss '{gloss}'. Possible shape mismatch.")
                                    raise e
                                
                                self.data.append(torch.tensor(padded_video, dtype=torch.float32))
                    except Exception as e:
                        logging.error(f"Error processing {file_path}: {e}")
    
    def _pad_video(self, video):
        
        num_existing_frames = len(video)
        if num_existing_frames >= MAX_FRAMES:
            return video[:MAX_FRAMES]  
        
        pad_frames = np.zeros((MAX_FRAMES - num_existing_frames, 49, 3), dtype=np.float32)
        padded_video = np.vstack((video, pad_frames))
        return padded_video

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].reshape(3, MAX_FRAMES, 49)


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=100, patience=10):
    logging.info("Starting Training...")
    model.train()
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}...")
        epoch_loss = 0.0

        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            refined_output = model(batch)
            loss = criterion(refined_output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 10 == 0:  
                logging.info(f"  Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} completed. Avg Train Loss: {avg_train_loss:.4f}")

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")

        current_lr = scheduler.optimizer.param_groups[0]['lr']
        logging.info(f"Learning Rate: {current_lr}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(STGCN_MODEL_PATH, "stgcn.pth")
            torch.save(model.state_dict(), model_save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}.")
                break

        scheduler.step()
        model.train()

    logging.info("Training Complete!")


def main():

    logging.info(f"Using device: {device}")
    os.makedirs(STGCN_MODEL_PATH, exist_ok=True)

    dataset = SignLanguageDataset(FINAL_JSONL_PATHS)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    
    logging.info(f"Dataset split: {len(train_set)} training, {len(val_set)} validation, {len(test_set)} test samples.")
    
    model = STGCN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    train_model(model, train_loader, val_loader, optimizer, criterion, scheduler)

if __name__ == "__main__":
    main()