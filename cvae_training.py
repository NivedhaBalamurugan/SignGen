import os
import glob
import orjson
import numpy as np
import torch
import logging
import gc
import psutil
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from architectures.cvae import ConditionalVAE, kl_divergence_loss, latent_classification_loss
from torch.optim.lr_scheduler import CosineAnnealingLR 
from torch.optim import AdamW
import torch.cuda.amp as amp
from utils.glove_utils import validate_word_embeddings
from utils.validation_utils import validate_data_shapes, validate_config
from torchsummary import summary
from config import *
import warnings
from tqdm import tqdm

# Constants for batch processing
FILES_PER_BATCH = 1
MAX_SAMPLES_PER_BATCH = 1000
MEMORY_THRESHOLD = 85
EVAL_FREQUENCY = 5  # Evaluate every 5 epochs

class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

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
        self.glove_embeddings = WORD_EMBEDDINGS
        if not self.glove_embeddings or not validate_word_embeddings(self.glove_embeddings, EMBEDDING_DIM):
            logging.error("Failed to validate word embeddings")
            return
        
        # Process files in batches
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
                with open(file_path, "rb") as f:
                    for line in f:
                        try:
                            item = orjson.loads(line)
                            for gloss, videos in item.items():
                                if gloss not in self.vocab:
                                    self.vocab[gloss] = len(self.vocab)
                                
                                # Process only a subset of videos if there are too many
                                for video in videos[:MAX_SAMPLES_PER_BATCH // len(item)]:
                                    padded_video = self._pad_video(video)
                                    
                                    try:
                                        padded_video = np.array(padded_video, dtype=np.float32)
                                    except Exception as e:
                                        logging.error(f"Error converting video to numpy array for gloss '{gloss}'. Possible shape mismatch.")
                                        continue
                                    
                                    # Add each video only once (fixed duplication)
                                    self.data.append((padded_video, gloss))
                        except Exception as e:
                            logging.error(f"Error processing line in {file_path}: {e}")
                logging.info(f"Completed reading from {file_path}")
            except Exception as e:
                logging.error(f"Error opening file {file_path}: {e}")
    
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


def train(model, train_loader, val_loader, device, num_epochs, lr, beta_start=0.1, beta_max=4.0, lambda_lc=0.1):
    logging.info("Starting training...")
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)    
    
    # Setup automatic mixed precision
    use_amp = torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast') and hasattr(torch.cuda.amp, 'GradScaler')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    autocast = torch.cuda.amp.autocast if use_amp else lambda: nullcontext()
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{num_epochs}...")
        epoch_loss = 0.0
        beta = min(beta_max, beta_start + (epoch / 20))  # Dynamically increase beta
        
        model.train()
        
        # Use tqdm for progress bar
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i, batch in enumerate(train_loader):
                # Check memory and clear if needed
                if i % 10 == 0:
                    check_memory()
                
                optimizer.zero_grad()
                video = batch['video'].to(device)
                condition = batch['condition'].to(device)

                if use_amp:
                    with autocast():
                        recon_video, mu, logvar, attn_weights, z_v = model(video, condition)
                        recon_loss = nn.functional.smooth_l1_loss(recon_video, video.view(video.size(0), video.size(1), -1))
                        kl_loss = kl_divergence_loss(mu, logvar).mean()
                        z_g = torch.normal(0, 1, size=z_v.shape).to(device)
                        lc_loss = latent_classification_loss(z_v, z_g, model.latent_classifier)
                        loss = recon_loss + beta * kl_loss + lambda_lc * lc_loss

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    recon_video, mu, logvar, attn_weights, z_v = model(video, condition)
                    recon_loss = nn.functional.smooth_l1_loss(recon_video, video.view(video.size(0), video.size(1), -1))
                    kl_loss = kl_divergence_loss(mu, logvar).mean()
                    z_g = torch.normal(0, 1, size=z_v.shape).to(device)
                    lc_loss = latent_classification_loss(z_v, z_g, model.latent_classifier)
                    loss = recon_loss + beta * kl_loss + lambda_lc * lc_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                epoch_loss += loss.item()
                
                # Only log detailed gradient stats occasionally
                if i % 50 == 0:
                    total_grad_norm = 0.0
                    max_grad = -float('inf')
                    min_grad = float('inf')
                    for param in model.parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            total_grad_norm += grad_norm
                            max_grad = max(max_grad, param.grad.abs().max().item())
                            min_grad = min(min_grad, param.grad.abs().min().item() if param.grad.abs().min().item() > 0 else min_grad)
                    logging.info(f"  Grad Norm: {total_grad_norm:.4f} | Max Grad: {max_grad:.4f} | Min Grad: {min_grad:.4f}")

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'recon': f'{recon_loss.item():.6f}',
                    'kl': f'{kl_loss.item():.6f}'
                })
                pbar.update(1)

        avg_train_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} completed. Avg Train Loss: {avg_train_loss:.6f}")
        
        # Only evaluate periodically to save time
        if (epoch + 1) % EVAL_FREQUENCY == 0 or epoch == 0 or epoch == num_epochs - 1:
            # Validation
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    video = batch['video'].to(device)
                    condition = batch['condition'].to(device)
                    recon_video, mu, logvar, attn_weights, z_v = model(video, condition)
                    z_g = torch.normal(0, 1, size=z_v.shape).to(device)
                    recon_loss = nn.functional.smooth_l1_loss(recon_video, video.view(video.size(0), video.size(1), -1))
                    kl_loss = kl_divergence_loss(mu, logvar).mean()
                    lc_loss = latent_classification_loss(z_v, z_g, model.latent_classifier)
                    loss = recon_loss + beta * kl_loss + lambda_lc * lc_loss
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            logging.info(f"Validation Loss: {avg_val_loss:.6f} | Beta: {beta:.6f}")
            
            # Save model and check early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                model_save_path = os.path.join(CVAE_MODEL_PATH, "cvae.pth")
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"Model saved to {model_save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info("Early stopping triggered.")
                    break
        
        scheduler.step()
        logging.info(f"Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# Simplified evaluation metrics
def evaluate_model(model, dataloader, device, max_samples=500):
    logging.info("Evaluating metrics (sample-based)...")
    model.eval()
    mse_values = []
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            video = batch['video'].to(device)
            condition = batch['condition'].to(device)
            recon_video, mu, logvar, _, _ = model(video, condition)
            
            # Compute MSE
            mse = torch.mean((video.view(video.size(0), video.size(1), -1) - recon_video) ** 2, dim=[1, 2])
            mse_values.append(mse.cpu())
            
            sample_count += video.size(0)
            if sample_count > max_samples:
                break
    
    # Aggregate results
    mse_values = torch.cat(mse_values, dim=0)
    avg_mse = torch.mean(mse_values).item()
    
    logging.info(f"Average MSE: {avg_mse:.8f}")
    return {"mse": avg_mse}


def main():
    if not validate_config():
        return

    os.makedirs(CVAE_MODEL_PATH, exist_ok=True)

    logging.info("Initializing dataset...")
    dataset = LandmarkDataset(FINAL_JSONL_PATHS)

    train_size = int(0.9 * len(dataset))  
    val_size = len(dataset) - train_size  

    # Create dataset splits
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Use more workers if available
    num_workers = min(4, os.cpu_count() or 1)
    logging.info(f"Using {num_workers} workers for data loading")
    
    # Initialize data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CVAE_BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CVAE_BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    # Initialize model
    model = ConditionalVAE(
        input_dim=CVAE_INPUT_DIM,
        hidden_dim=CVAE_HIDDEN_DIM,
        latent_dim=CVAE_LATENT_DIM,
        cond_dim=EMBEDDING_DIM,
        output_dim=CVAE_INPUT_DIM,
        seq_len=MAX_FRAMES
    ).to(device)

    logging.info("Model initialized. Starting training...")

    # Train the model
    train(model, train_loader, val_loader, device, num_epochs=80, lr=0.005, beta_start=0.1, beta_max=4.0, lambda_lc=0.1)

    logging.info("Training completed. Running final evaluation...")
    
    # Final evaluation
    evaluate_model(model, val_loader, device)

if __name__ == "__main__":
    main()