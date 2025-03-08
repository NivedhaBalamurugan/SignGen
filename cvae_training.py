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


# Constants for batch processing
FILES_PER_BATCH = 1
MAX_SAMPLES_PER_BATCH = 1000
MEMORY_THRESHOLD = 85
EVAL_FREQUENCY = 10
PREFETCH_FACTOR = 2
MEMORY_THRESHOLD = 85

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
        self.glove_embeddings = load_word_embeddings()
        if not self.glove_embeddings or not validate_word_embeddings(self.glove_embeddings, EMBEDDING_DIM):
            logging.error("Failed to validate word embeddings")
            return
        
        # Process files in batches
        file_paths = sorted(glob.glob(file_paths))
        if not file_paths:
            logging.error(f"No files found matching pattern: {file_paths}")
            return
            
        # Process files in parallel if possible
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
                video_dict = {}  # Cache to avoid duplicate processing
                
                with open(file_path, "rb") as f:
                    for line in f:
                        try:
                            item = orjson.loads(line)  # orjson is faster than json
                            for gloss, videos in item.items():
                                if gloss not in self.vocab:
                                    self.vocab[gloss] = len(self.vocab)
                                
                                # Remove the hard limit on videos per gloss
                                for video in videos:
                                    # Convert once then reuse
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
                                        
                                    # Add to dataset only once
                                    self.data.append((padded_video, gloss))
                        except Exception as e:
                            logging.error(f"Error processing line in {file_path}: {e}")
                            continue
                logging.info(f"Completed reading from {file_path}")
            except Exception as e:
                logging.error(f"Error opening file {file_path}: {e}")
                

    def __len__(self):
        return len(self.data)
    
    # Cache frequently accessed items
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


def train(model, train_loader, val_loader, device, num_epochs=100, lr=1e-3, beta_start=0.1, beta_max=1.0, lambda_lc=0):
    logging.info("Starting training...")
    
    # Use AdamW with a slightly higher learning rate
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)      
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Initial restart period
        T_mult=2,  # Multiplicative factor for restart
        eta_min=1e-5  # Minimum learning rate
    )
        # Setup automatic mixed precision - always use if available
    use_amp = torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast') and hasattr(torch.cuda.amp, 'GradScaler')
    scaler = amp.GradScaler() if use_amp else None
    autocast = amp.autocast if use_amp else lambda: nullcontext()
    
    best_val_loss = float('inf')
    max_patience = 15  # Reduced from 15
    patience_counter = 0
    
    
    # Compile model if using PyTorch 2.0+
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            model = torch.compile(model)
            logging.info("Model successfully compiled with torch.compile()")
        except Exception as e:
            logging.warning(f"Could not compile model: {e}")

    for epoch in range(num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{num_epochs}...")
        epoch_loss = 0.0
        warmup_epochs = 50 
        if epoch < warmup_epochs:
            beta = beta_start * (epoch / warmup_epochs)
        else:
            beta = beta_max

        
        model.train()
        total_train_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_lc_loss = 0
        
        # Use tqdm for progress bar
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i, batch in enumerate(train_loader):
                # Reduce memory check frequency
                if i % 50 == 0:
                    check_memory()
                
                optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
                video = batch['video'].to(device, non_blocking=True)
                condition = batch['condition'].to(device, non_blocking=True)

                # batch_size = video.size(0)
                
                if use_amp:
                    with autocast():
                        recon_video, mu, logvar, attn_weights, z_v = model(video, condition)
                        
                        z_g = torch.randn_like(z_v)
                            
                        recon_loss = adaptive_reconstruction_loss(recon_video, video)
                        kl_loss = improved_kl_divergence_loss(mu, logvar, beta)
                        lc_loss = adaptive_latent_classification_loss(z_v, z_g, model.latent_classifier)
                        loss = recon_loss + kl_loss + lambda_lc * lc_loss

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    recon_video, mu, logvar, attn_weights, z_v = model(video, condition)
                    
                    z_g = torch.randn_like(z_v)
                        
                    recon_loss = adaptive_reconstruction_loss(recon_video, video)
                    kl_loss = improved_kl_divergence_loss(mu, logvar, beta)
                    lc_loss = adaptive_latent_classification_loss(z_v, z_g, model.latent_classifier)
                    loss = recon_loss + kl_loss + lambda_lc * lc_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_train_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_lc_loss += lc_loss.item()

                pbar.update(1)
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}'
                })
                

            scheduler.step()
                
        # Validation
        total_val_loss = 0.0
        model.eval()
        
        # Use torch.no_grad for validation (faster)
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                video = batch['video'].to(device, non_blocking=True)
                condition = batch['condition'].to(device, non_blocking=True)
                
                # batch_size = video.size(0)
                
                # Even in evaluation, we can use autocast for speed
                if use_amp:
                    with autocast():
                        recon_video, mu, logvar, attn_weights, z_v = model(video, condition)
                        z_g = torch.randn_like(z_v)
                        recon_loss = adaptive_reconstruction_loss(recon_video, video)
                        kl_loss = improved_kl_divergence_loss(mu, logvar, beta)
                        lc_loss = adaptive_latent_classification_loss(z_v, z_g, model.latent_classifier)
                        
                        val_loss = recon_loss + kl_loss + lambda_lc * lc_loss
                        total_val_loss += val_loss.item()
                else:
                    recon_video, mu, logvar, attn_weights, z_v = model(video, condition)
                    z_g = torch.randn_like(z_v)
                    recon_loss = adaptive_reconstruction_loss(recon_video, video)
                    kl_loss = improved_kl_divergence_loss(mu, logvar, beta)
                    lc_loss = adaptive_latent_classification_loss(z_v, z_g, model.latent_classifier)
                    
                    val_loss = recon_loss + kl_loss + lambda_lc * lc_loss
                    total_val_loss += val_loss.item()
                    

        avg_train_loss = total_train_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        avg_lc_loss = total_lc_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        logging.info(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.8f}")
        logging.info(f"Recon loss {avg_recon_loss:.8f} , Kl loss {avg_kl_loss:.8f} , LC loss {avg_lc_loss:.8f}")
        logging.info(f"Val Loss {avg_val_loss:.8f} , Beta {beta:.8f}", )
        
        model_save_path = os.path.join(CVAE_MODEL_PATH, f"cvae{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"Model saved to {model_save_path}")
        
        # Save model and check early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            logging.info("Early stopping triggered.")
            break
        
        # Force garbage collection
        if epoch % 2 == 0:  # Less frequent GC
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None




# Simplified evaluation metrics with sampling
def evaluate_model(model, dataloader, device, max_samples=500):
    logging.info("Evaluating metrics (sample-based)...")
    model.eval()
    mse_values = []
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            video = batch['video'].to(device, non_blocking=True)
            condition = batch['condition'].to(device, non_blocking=True)
            
            # Use autocast for evaluation too
            if torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast'):
                with torch.cuda.amp.autocast():
                    recon_video, mu, logvar, _, _ = model(video, condition)
                    # Compute MSE
                    mse = torch.mean((video.view(video.size(0), video.size(1), -1) - recon_video) ** 2, dim=[1, 2])
            else:
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
    num_workers = min(8, os.cpu_count() or 1)  # Increased from 4
    logging.info(f"Using {num_workers} workers for data loading")
    
    # Initialize data loaders with performance optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CVAE_BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CVAE_BATCH_SIZE * 2,  # Larger batch size for validation
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

    # Move to GPU before training if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model and move to device
    model = ConditionalVAE(
        input_dim=CVAE_INPUT_DIM,
        hidden_dim=CVAE_HIDDEN_DIM,
        latent_dim=CVAE_LATENT_DIM,
        cond_dim=EMBEDDING_DIM,
        output_dim=CVAE_INPUT_DIM,
        seq_len=MAX_FRAMES
    ).to(device)

    # Log configuration
    logging.info(f"Training on device: {device}")
    logging.info(f"Using mixed precision: {torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast')}")
    logging.info(f"Dataset size: {len(dataset)}, Train: {train_size}, Val: {val_size}")
    logging.info("Model initialized. Starting training...")

    # Train the model with a slightly higher learning rate
    train(model, train_loader, val_loader, device, num_epochs=100, lr=1e-3)

    logging.info("Training completed. Running final evaluation...")
    
    # input_shape = (CVAE_BATCH_SIZE, MAX_FRAMES, CVAE_INPUT_DIM)
    # cond_shape = (CVAE_BATCH_SIZE, EMBEDDING_DIM)
    # summary(model, input_size=[input_shape, cond_shape])

    evaluate_model(model, val_loader, device)

if __name__ == "__main__":
    setup_logging("cvae_training")
    main()