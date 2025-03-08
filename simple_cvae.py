import torch 
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import traceback
import warnings
import psutil
import logging
import gc
import glob
import tqdm
import orjson
from functools import lru_cache
from config import *
from utils.data_utils import load_word_embeddings

# Hyperparameters
batch_size = 32
learning_rate = 1e-3
max_epoch = 100
load_epoch = -1
generate = True
NUM_CLASSES = 30  # Number of sign language classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILES_PER_BATCH = 1
MAX_SAMPLES_PER_BATCH = 1000
MEMORY_THRESHOLD = 85
EVAL_FREQUENCY = 10
PREFETCH_FACTOR = 2


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

class SignLanguageModel(nn.Module):
    def __init__(self, 
                 input_dim=NUM_COORDINATES,  # 3D coordinates per joint
                 num_joints=NUM_JOINTS, 
                 num_frames=MAX_FRAMES, 
                 latent_size=64,
                 embedding_dim=EMBEDDING_DIM,
                 num_classes=30):
        super(SignLanguageModel, self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.num_joints = num_joints
        self.num_frames = num_frames

        # Encoder BiLSTM Layers
        self.bilstm = nn.LSTM(
            input_size=input_dim,  # 3D coordinates
            hidden_size=64,  # Increased hidden size
            num_layers=2,  # Two-layer BiLSTM
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layers after BiLSTM
        self.fc_encoder = nn.Sequential(
            nn.Linear(64 * 2 * num_joints, 256),  # Doubled size to capture more features
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Latent space layers
        self.mu = nn.Linear(256 + embedding_dim, latent_size)
        self.logvar = nn.Linear(256 + embedding_dim, latent_size)

        # Decoder Layers
        self.fc_decoder_input = nn.Sequential(
            nn.Linear(latent_size + embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc_decoder_output = nn.Sequential(
            nn.Linear(256, 64 * 2 * num_joints),
            nn.BatchNorm1d(64 * 2 * num_joints),
            nn.ReLU()
        )

        # BiLSTM-like Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Final projection to original input dimension
        self.output_proj = nn.Linear(64 * 2, input_dim)

    def encoder(self, x, condition):
        B, F, J, C = x.shape
        x_reshaped = x.view(B * J, F, C)
        
        # Process each joint separately to reduce memory usage
        lstm_out_list = []
        for joint_idx in range(x_reshaped.shape[0]):
            single_joint = x_reshaped[joint_idx:joint_idx+1]
            single_lstm_out, _ = self.bilstm(single_joint)
            lstm_out_list.append(single_lstm_out.mean(dim=1))
        
        lstm_out = torch.stack(lstm_out_list)
        lstm_out = lstm_out.view(B, -1)
        
        fc_features = self.fc_encoder(lstm_out)
        
        # Concatenate condition vector with fc_features
        combined_features = torch.cat([fc_features, condition], dim=1)
        
        mu = self.mu(combined_features)
        logvar = self.logvar(combined_features)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def decoder(self, z):
        B = z.shape[0]
        
        decoder_input = self.fc_decoder_input(z)
        decoder_features = self.fc_decoder_output(decoder_input)
        decoder_features = decoder_features.view(B, self.num_joints, 64 * 2)
        
        dummy_input = torch.zeros(B, self.num_frames, self.num_joints, 3).to(z.device)
        
        decoded_sequence = []
        for joint in range(self.num_joints):
            joint_features = decoder_features[:, joint, :]
            joint_features = joint_features.unsqueeze(1).repeat(1, self.num_frames, 1)
            
            joint_decoded, _ = self.decoder_lstm(dummy_input[:, :, joint, :])
            joint_coords = self.output_proj(joint_decoded)
            decoded_sequence.append(joint_coords)
        
        decoded_sequence = torch.stack(decoded_sequence, dim=2)
        
        return decoded_sequence

    def forward(self, x, condition):
        mu, logvar = self.encoder(x, condition)
        z = self.reparameterize(mu, logvar)
        z = torch.cat((z, condition), dim=1)
        pred = self.decoder(z)
        return pred, mu, logvar


def plot(epoch, pred, condition, name='test_'):
    """
    Plot reconstructed sequences (simplified for demonstration)
    """
    if not os.path.isdir('./images'):
        os.mkdir('./images')
    
    fig = plt.figure(figsize=(16,16))
    for i in range(min(6, pred.shape[0])):
        ax = fig.add_subplot(3,2,i+1)
        # Plot first joint's first coordinate as an example
        ax.plot(pred[i,0,:,0].cpu().numpy())
        ax.set_title(f"Condition: {condition[i]}")
        ax.axis('off')
    
    plt.savefig(f"./images/{name}epoch_{epoch}.jpg")
    plt.close()

def loss_function(x, pred, mu, logvar):
    """
    Compute reconstruction and KL divergence losses
    """
    recon_loss = F.mse_loss(pred, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kld

def train(epoch, model, train_loader, optimizer):
    """
    Training function
    """
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    
    model.train()
    for i, batch in enumerate(train_loader):
        try:
            x = batch['video']
            condition = batch['condition']

            optimizer.zero_grad()   
            pred, mu, logvar = model(x.to(device), condition.to(device))
            
            recon_loss, kld = loss_function(x.to(device), pred, mu, logvar)
            loss = recon_loss + kld
            loss.backward()
            optimizer.step()

            total_loss += loss.cpu().data.numpy() * x.shape[0]
            reconstruction_loss += recon_loss.cpu().data.numpy() * x.shape[0]
            kld_loss += kld.cpu().data.numpy() * x.shape[0]
            
            # Print gradient information (optional)
            if i == 0:
                print("Gradients:")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: {param.grad.abs().mean().item()}")
        except Exception as e:
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue
    
    reconstruction_loss /= len(train_loader.dataset)
    kld_loss /= len(train_loader.dataset)
    total_loss /= len(train_loader.dataset)
    return total_loss, kld_loss, reconstruction_loss

def val(epoch, model, test_loader):
    """
    Testing/Validation function
    """
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            try:
                x = batch['video']
                condition = batch['condition']

                pred, mu, logvar = model(x.to(device), condition.to(device))
                recon_loss, kld = loss_function(x.to(device), pred, mu, logvar)
                loss = recon_loss + kld

                total_loss += loss.cpu().data.numpy() * x.shape[0]
                reconstruction_loss += recon_loss.cpu().data.numpy() * x.shape[0]
                kld_loss += kld.cpu().data.numpy() * x.shape[0]
                
                # Plot first batch of reconstructions
                if i == 0:
                    plot(epoch, pred.cpu(), condition.cpu())
            except Exception as e:
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue
    
    reconstruction_loss /= len(test_loader.dataset)
    kld_loss /= len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, kld_loss, reconstruction_loss

def generate_sequence(epoch, z, condition, model):
    """
    Generate sequences from random latent vectors
    """
    with torch.no_grad():
        pred = model.decoder(torch.cat((z.to(device), condition.float().to(device)), dim=1))
        plot(epoch, pred.cpu(), condition.cpu(), name='Eval_')
        print("Sequences Plotted")


class SignLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, transform=None):
        self.data = []
        self.vocab = {}
        self.transform = transform
        self.glove_embeddings = load_word_embeddings()
        
        file_paths = sorted(glob.glob(file_paths))
        if not file_paths:
            logging.error(f"No files found matching pattern: {file_paths}")
            return
        
        # Process files in parallel if possible
        num_batches = -(-len(file_paths)//FILES_PER_BATCH)
        with tqdm.tqdm(total=num_batches, desc="Processing files") as pbar:
            for i in range(0, len(file_paths), FILES_PER_BATCH):
                file_batch = file_paths[i:i + FILES_PER_BATCH]
                self._process_file_batch(file_batch)
                check_memory()
                pbar.update(1)
                        
        logging.info(f"Dataset processing complete. Vocabulary size: {len(self.vocab)}, data size: {len(self.data)}")

    
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
                                        padded_video = self._pad_video(video)
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
                                    self.data.append((padded_video, gloss))
                                    self.data.append((padded_video, gloss))
                        except Exception as e:
                            logging.error(f"Error processing line in {file_path}: {e}")
                            continue
                logging.info(f"Completed reading from {file_path}")
            except Exception as e:
                logging.error(f"Error opening file {file_path}: {e}")


    def _pad_video(self, video):
        num_existing_frames = len(video)
        if num_existing_frames >= MAX_FRAMES:
            return video[:MAX_FRAMES]
        
        padded_video = np.zeros((MAX_FRAMES, 49, 3), dtype=np.float32)
        padded_video[:num_existing_frames] = np.array(video[:num_existing_frames])
        return padded_video

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
    


def load_data():
    """
    Load sign language dataset
    """

    # Create dataset
    full_dataset = SignLanguageDataset(FINAL_JSONL_PATHS)
    
    # Split dataset
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    num_workers = min(8, os.cpu_count() or 1)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader

def save_model(model, epoch):
    """
    Save model checkpoint
    """
    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")
    file_name = f'./checkpoints/model_{epoch}.pt'
    torch.save(model.state_dict(), file_name)

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load data
    train_loader, val_loader = load_data()
    logging.info("Dataloader created")

    # Initialize model
    model = SignLanguageModel(
        input_dim=3, 
        num_joints=49, 
        num_frames=233, 
        latent_size=64,
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES
    ).to(device)
    logging.info("Model created")
    
    # Load pre-trained model if specified
    if load_epoch > 0:
        model.load_state_dict(torch.load(f'./checkpoints/model_{load_epoch}.pt', map_location=device))
        logging.info(f"Model {load_epoch} loaded")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    # Tracking losses
    train_loss_list = []
    val_loss_list = []

    # Training loop
    for i in range(load_epoch+1, max_epoch):
        # Train
        model.train()
        train_total, train_kld, train_loss = train(i, model, train_loader, optimizer)
        
        # Validate
        with torch.no_grad():
            model.eval()
            val_total, val_kld, val_loss = val(i, model, val_loader)
        
        # Generate sequences
        if generate:
            z = torch.randn(6, 64).to(device)  # Random latent vectors
            
            # Get random condition vectors from the validation dataset
            condition_samples = [val_loader.dataset[np.random.randint(len(val_loader.dataset))]['condition'] for _ in range(6)]
            condition = torch.stack(condition_samples).to(device)
            
            generate_sequence(i, z, condition, model)
        
        # Print epoch summary
        logging.info(f"Epoch: {i}/{max_epoch}")
        logging.info(f"Train loss: {train_total}, Train KLD: {train_kld}, Train Reconstruction Loss: {train_loss}")
        logging.info(f"Val loss: {val_total}, Val KLD: {val_kld}, Val Reconstruction Loss: {val_loss}")

        # Save model
        save_model(model, i)
        
        # Store losses
        train_loss_list.append([train_total, train_kld, train_loss])
        val_loss_list.append([val_total, val_kld, val_loss])
        
        # Save loss history
        np.save("train_loss", np.array(train_loss_list))
        np.save("val_loss", np.array(val_loss_list))

if __name__ == "__main__":
    main()