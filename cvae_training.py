import os
import glob
import orjson
import numpy as np
import torch
import gc
import psutil
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn.functional as F
from architectures.cvae import ConditionalVAE, improved_kl_divergence_loss, adaptive_reconstruction_loss, adaptive_latent_classification_loss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from torch.optim import AdamW
import torch.cuda.amp as amp
from utils.data_utils import load_word_embeddings
from utils.glove_utils import validate_word_embeddings
from utils.validation_utils import validate_data_shapes, validate_config
from config import *
import warnings
from tqdm import tqdm
from functools import lru_cache
from torchsummary import summary


FILES_PER_BATCH = 1
MAX_SAMPLES_PER_BATCH = 1000
MEMORY_THRESHOLD = 85
EVAL_FREQUENCY = 10
PREFETCH_FACTOR = 2
MEMORY_THRESHOLD = 85



warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_memory():
    memory_percent = psutil.Process().memory_percent()
    if memory_percent > MEMORY_THRESHOLD:
        logging.warning(
            f"Memory usage high ({memory_percent:.1f}%). Clearing memory...")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return True
    return False


class LandmarkDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        logging.info(f"Loading dataset from {file_paths}...")
        self.transform = transform
        self.data = []
        self.vocab = {}
        self.glove_embeddings = load_word_embeddings()
        if not self.glove_embeddings or not validate_word_embeddings(self.glove_embeddings, EMBEDDING_DIM):
            logging.error("Failed to validate word embeddings")
            return
        
        file_paths = sorted(glob.glob(file_paths))
        if not file_paths:
            logging.error(f"No files found matching pattern: {file_paths}")
            return
            
        num_batches = -(-len(file_paths)//FILES_PER_BATCH)
        with tqdm(total=num_batches, desc="Processing files") as pbar:
            for i in range(0, len(file_paths), FILES_PER_BATCH):
                file_batch = file_paths[i:i + FILES_PER_BATCH]
                self._process_file_batch(file_batch)
                check_memory()
                pbar.update(1)
                
        logging.info(f"Dataset processing complete. Max sequence length: {MAX_FRAMES}, Vocabulary size: {len(self.vocab)}, data size: {len(self.data)}")

    def _process_file_batch(self, file_paths):
        for file_path in file_paths:
            try:
                video_dict = {}  
                
                with open(file_path, "rb") as f:
                    for line in f:
                        try:
                            item = orjson.loads(line)  
                            for gloss, videos in item.items():
                                if gloss not in self.vocab:
                                    self.vocab[gloss] = len(self.vocab)
                                
                                for video in videos:
                                    video_hash = hash(str(video))
                                    if video_hash not in video_dict:
                                        padded_video = video
                                        try:
                                            padded_video = np.array(padded_video, dtype=np.float32)
                                            video_dict[video_hash] = padded_video
                                        except Exception as e:
                                            logging.error(f"Error converting video for gloss '{gloss}'")
                                            continue
                                    else:
                                        padded_video = video_dict[video_hash]
                                        
                                    self.data.append((padded_video, gloss))
                        except Exception as e:
                            logging.error(f"Error processing line in {file_path}: {e}")
                            continue
                logging.info(f"Completed reading from {file_path}")
            except Exception as e:
                logging.error(f"Error opening file {file_path}: {e}")
                

    def __len__(self):
        return len(self.data)
    
    @lru_cache(maxsize=512)
    def _get_cond_vector(self, gloss):
        cond_vector = self.glove_embeddings.get(gloss, np.zeros(EMBEDDING_DIM))
        return torch.tensor(cond_vector, dtype=torch.float32)
    
    def __getitem__(self, idx):
        video, gloss = self.data[idx]
        video_tensor = torch.tensor(video)
        cond_vector = self._get_cond_vector(gloss)
        
        sample = {'video': video_tensor, 'condition': cond_vector}
        if self.transform:
            sample = self.transform(sample)
        return sample


def train(model, train_loader, val_loader, device, num_epochs, lr, beta_start=0.1, beta_max=4.0, lambda_lc=0.1):
    logging.info("Starting training...")
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{num_epochs}...")
        epoch_loss = 0.0
        beta = min(beta_max, beta_start + (epoch / 10)) 

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            video = batch['video'].to(device)
            condition = batch['condition'].to(device)

            if torch.cuda.is_available():
                with autocast():
                    recon_video, mu, logvar, attn_weights, z_v = model(video, condition)
                    
                    recon_loss = nn.functional.smooth_l1_loss(recon_video, video.view(video.size(0), video.size(1), -1))
                    
                    kl_loss = kl_divergence_loss(mu, logvar).mean()

                    z_g = torch.normal(0, 1, size=z_v.shape).to(device)
                    lc_loss = latent_classification_loss(z_v, z_g, model.latent_classifier)

                    loss = recon_loss + beta * kl_loss + lambda_lc * lc_loss
            else:
                recon_video, mu, logvar, attn_weights, z_v = model(video, condition)
                recon_loss = nn.functional.smooth_l1_loss(recon_video, video.view(video.size(0), video.size(1), -1))
                kl_loss = kl_divergence_loss(mu, logvar).mean()
                z_g = torch.normal(0, 1, size=z_v.shape).to(device)
                lc_loss = latent_classification_loss(z_v, z_g, model.latent_classifier)
                loss = recon_loss + beta * kl_loss + lambda_lc * lc_loss

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if i % 10 == 0:
                logging.info(f"  Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f} | LC: {lc_loss.item():.4f}")

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(device)
                condition = batch['condition'].to(device)
                recon_video, mu, logvar, attn_weights, z_v = model(video, condition)
                z_g = torch.normal(0, 1, size=z_v.shape).to(device)
                recon_loss = nn.functional.smooth_l1_loss(recon_video, video.view(video.size(0), video.size(1), -1))
                kl_loss = kl_divergence_loss(mu, logvar).mean()
                lc_loss = latent_classification_loss(z_v, z_g, model.latent_classifier)
                loss = recon_loss + beta * kl_loss + lambda_lc * lc_loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(avg_val_loss)

        logging.info(f"Epoch {epoch+1} completed. Avg Loss: {epoch_loss/len(train_loader.dataset):.4f}")
        logging.info(f"Validation Loss: {avg_val_loss:.4f} | Beta: {beta:.2f}")
        logging.info(f"Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model_save_path = os.path.join(CVAE_MODEL_PATH, "cvae.pth")
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break

def compute_mse(original, reconstructed):
    return torch.mean((original - reconstructed) ** 2).item()

def compute_ssim(original, reconstructed):
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    ssim_val = 0
    for i in range(original.shape[0]):  
        ssim_val += ssim(original[i], reconstructed[i], multichannel=True)
    return ssim_val / original.shape[0]

def evaluate_model(model, dataloader, device):

    logging.info("Evaluating metris")
    model.eval()
    mse_values, ssim_values = [], []
    latent_vectors = []
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            condition = batch['condition'].to(device)
            recon_video, mu, logvar, _, z = model(video, condition)
            
            mse = compute_mse(video, recon_video)
            ssim_val = compute_ssim(video, recon_video)
            mse_values.append(mse)
            ssim_values.append(ssim_val)
            
            latent_vectors.append(z)
    
    avg_mse = np.mean(mse_values)
    avg_ssim = np.mean(ssim_values)
    latent_vectors = torch.cat(latent_vectors, dim=0)
    
    logging.info(f"Average MSE: {avg_mse:.4f}")
    logging.info(f"Average SSIM: {avg_ssim:.4f}")
    



def main():
    if not validate_config():
        return

    os.makedirs(CVAE_MODEL_PATH, exist_ok=True)

    logging.info("Initializing dataset...")
    dataset = LandmarkDataset(FINAL_JSONL_PATHS)

    logging.info(f"Max frame count (sequence length): {MAX_FRAMES}")
    logging.info("Vocabulary:", dataset.vocab)

    logging.info("Initializing model...")

    train_size = int(0.9 * len(dataset))  
    val_size = len(dataset) - train_size  

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=CVAE_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CVAE_BATCH_SIZE, shuffle=False, num_workers=0)

    model = ConditionalVAE(
        input_dim=CVAE_INPUT_DIM,
        hidden_dim=CVAE_HIDDEN_DIM,
        latent_dim=CVAE_LATENT_DIM,
        cond_dim=EMBEDDING_DIM,
        output_dim=CVAE_INPUT_DIM,
        seq_len=MAX_FRAMES
    ).to(device)

    logging.info("Model initialized. Starting training...")

    input_shape = (CVAE_BATCH_SIZE, MAX_FRAMES, CVAE_INPUT_DIM)
    cond_shape = (CVAE_BATCH_SIZE, EMBEDDING_DIM)
    summary(model, input_size=[input_shape, cond_shape])


    train(model, train_loader, val_loader, device, num_epochs=100, lr=0.001, beta_start=0.01, beta_max=4.0, lambda_lc=0.01)

    logging.info("Model saved successfully ")

    
    input_shape = (CVAE_BATCH_SIZE, MAX_FRAMES, CVAE_INPUT_DIM)
    cond_shape = (CVAE_BATCH_SIZE, EMBEDDING_DIM)
    summary(model, input_size=[input_shape, cond_shape])

    evaluate_model(model, val_loader, device)

if __name__ == "__main__":
    main()