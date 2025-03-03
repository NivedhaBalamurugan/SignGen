import os
import glob
import orjson
import numpy as np
import torch
import logging
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from architectures.cvae import ConditionalVAE, kl_divergence_loss, latent_classification_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from utils.glove_utils import validate_word_embeddings
from utils.validation_utils import validate_data_shapes, validate_config
from torchsummary import summary
from config import *
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LandmarkDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        logging.info(f"Loading dataset from {file_paths}...")
        self.transform = transform
        self.data = []
        self.vocab = {}
        self._load_process_data(file_paths)
        logging.info(f"Dataset processing complete. Max sequence length: {MAX_FRAMES}, Vocabulary size: {len(self.vocab)}, data size: {len(self.data)}")
        self.glove_embeddings = WORD_EMBEDDINGS
        if not self.glove_embeddings or not validate_word_embeddings(self.glove_embeddings, EMBEDDING_DIM):
            return

    def _load_process_data(self, file_paths):        
        
        file_paths = glob.glob(file_paths)
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                for line in f:
                    try:
                        item = orjson.loads(line)
                        for gloss, videos in item.items():
                            if gloss not in self.vocab:
                                self.vocab[gloss] = len(self.vocab)
                            # cond_vector = WORD_EMBEDDINGS.get(asl_word, np.zeros(EMBEDDING_DIM))
                            for video in videos:
                                padded_video = self._pad_video(video)
                                
                                try:
                                    padded_video = np.array(padded_video, dtype=np.float32)
                                except Exception as e:
                                    logging.error(f"Error converting video to numpy array for gloss '{gloss}'. Possible shape mismatch.")
                                    raise e
                                
                                self.data.append((padded_video, gloss))
                                self.data.append((padded_video, gloss))
                    except Exception as e:
                        logging.error(f"Error processing {file_path}: {e}")
            logging.info(f"Completed reading from {file_path}")
    
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
        video, gloss = self.data[idx]
        video_tensor = torch.tensor(video)

        cond_vector = self.glove_embeddings.get(gloss, np.zeros(EMBEDDING_DIM))
        cond_vector = torch.tensor(cond_vector, dtype=torch.float32)
        
        sample = {'video': video_tensor, 'condition': cond_vector}
        if self.transform:
            sample = self.transform(sample)
        return sample


def train(model, train_loader, val_loader, device, num_epochs, lr=0.001, beta_start=0.1, beta_max=4.0, lambda_lc=0.01):
    logging.info("Starting training...")
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # Added weight decay
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    warmup_epochs = 20  # KL warmup period

    for epoch in range(num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{num_epochs}...")
        epoch_loss = 0.0

        # KL Warmup
        if epoch < warmup_epochs:
            beta = beta_start * (epoch / warmup_epochs)  # Gradually increase beta
        else:
            beta = beta_max

        # Learning Rate Warmup
        if epoch < 10:  # Warmup for 10 epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * ((epoch + 1) / 10)

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            video = batch['video'].to(device)
            condition = batch['condition'].to(device)

            if torch.cuda.is_available():
                with autocast():
                    recon_video, mu, logvar, attn_weights, z_v = model(video, condition)
                    
                    # Smooth L1 Loss instead of MSE
                    recon_loss = nn.functional.smooth_l1_loss(recon_video, video.view(video.size(0), video.size(1), -1))
                    
                    # Compute KL Loss (Ensure mean over batch) and enforce minimum KL loss
                    kl_loss = kl_divergence_loss(mu, logvar).mean() + 1e-4  # Add 0.0001 to prevent collapse

                    # Generate a normal latent vector for classification loss
                    z_g = torch.normal(0, 1, size=z_v.shape).to(device)
                    lc_loss = latent_classification_loss(z_v, z_g, model.latent_classifier)

                    # Final Loss Calculation
                    loss = recon_loss + beta * kl_loss + lambda_lc * lc_loss
            else:
                recon_video, mu, logvar, attn_weights, z_v = model(video, condition)
                recon_loss = nn.functional.smooth_l1_loss(recon_video, video.view(video.size(0), video.size(1), -1))
                kl_loss = kl_divergence_loss(mu, logvar).mean() + 1e-4  # Add 0.0001 to prevent collapse
                z_g = torch.normal(0, 1, size=z_v.shape).to(device)
                lc_loss = latent_classification_loss(z_v, z_g, model.latent_classifier)
                loss = recon_loss + beta * kl_loss + lambda_lc * lc_loss

            # Backpropagation
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjusted gradient clipping
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if i % 10 == 0:
                logging.info(f"  Batch {i}/{len(train_loader)} | Loss: {loss.item():.8f} | Recon: {recon_loss.item():.8f} | KL: {kl_loss.item():.8f} | LC: {lc_loss.item():.8f}")

        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(device)
                condition = batch['condition'].to(device)
                recon_video, mu, logvar, attn_weights, z_v = model(video, condition)
                z_g = torch.normal(0, 1, size=z_v.shape).to(device)
                recon_loss = nn.functional.smooth_l1_loss(recon_video, video.view(video.size(0), video.size(1), -1))
                kl_loss = kl_divergence_loss(mu, logvar).mean() + 1e-4  # Add 0.0001 to prevent collapse
                lc_loss = latent_classification_loss(z_v, z_g, model.latent_classifier)
                loss = recon_loss + beta * kl_loss + lambda_lc * lc_loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(avg_val_loss)

        logging.info(f"Epoch {epoch+1} completed. Avg Loss: {epoch_loss/len(train_loader.dataset):.8f}")
        logging.info(f"Validation Loss: {avg_val_loss:.8f} | Beta: {beta:.8f}")
        logging.info(f"Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")

        # Early stopping
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

            

##eval metrics

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def compute_mse(original, reconstructed):
    return torch.mean((original - reconstructed) ** 2).item()

def compute_mae(original, reconstructed):
    return torch.mean(torch.abs(original - reconstructed)).item()

def compute_psnr(mse, max_val=1.0):
    return 10 * torch.log10(max_val ** 2 / mse).item()

def compute_ssim(original, reconstructed):
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    ssim_val = 0
    for i in range(original.shape[0]):  # Loop over batch
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
            
            # Compute metrics
            mse = compute_mse(video, recon_video)
            ssim_val = compute_ssim(video, recon_video)
            mse_values.append(mse)
            ssim_values.append(ssim_val)
            
            # Collect latent vectors
            latent_vectors.append(z)
    
    # Aggregate results
    avg_mse = np.mean(mse_values)
    avg_ssim = np.mean(ssim_values)
    latent_vectors = torch.cat(latent_vectors, dim=0)
    
    logging.info(f"Average MSE: {avg_mse:.8f}")
    logging.info(f"Average SSIM: {avg_ssim:.8f}")
    



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

    train(model, train_loader, val_loader, device, num_epochs=100, lr=0.001, beta_start=0.1, beta_max=4.0, lambda_lc=0.01)

    logging.info("Model saved successfully ")

    
    # input_shape = (CVAE_BATCH_SIZE, MAX_FRAMES, CVAE_INPUT_DIM)
    # cond_shape = (CVAE_BATCH_SIZE, EMBEDDING_DIM)
    # summary(model, input_size=[input_shape, cond_shape])

    evaluate_model(model, val_loader, device)

if __name__ == "__main__":
    main()
