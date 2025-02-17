import os
import gzip
import orjson
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

def load_glove_embeddings(filepath, embedding_dim=50):
    print(f"Loading GloVe embeddings from {filepath}...")
    embedding_dict = {}
    
    with open(filepath, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embedding_dict[word] = vector
    
    print(f"Loaded {len(embedding_dict)} word embeddings.")
    return embedding_dict

class LandmarkDataset(Dataset):
    def __init__(self, file_path, glove_path='Dataset/Glove/glove.6B.50d.txt', embedding_dim=50, transform=None):
        print(f"Loading dataset from {file_path}...")
        self.transform = transform
        self.data = []
        self.vocab = {}
        self.max_frames = 0
        self.expected_frame_length = 49 * 3
        self.embedding_dim = embedding_dim

        self.glove_embeddings = load_glove_embeddings(glove_path, embedding_dim)
        
        raw_data = self._load_data(file_path)
        print(f"Loaded {len(raw_data)} gloss entries.")
        self._process_data(raw_data)
        print(f"Dataset processing complete. Max sequence length: {self.max_frames}, Vocabulary size: {len(self.vocab)}")

    def _load_data(self, file_path):
        raw_data = defaultdict(list)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist")
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
                
                try:
                    padded_video = np.array(padded_video, dtype=np.float32)
                except Exception as e:
                    print(f"Error converting video to numpy array for gloss '{gloss}'. Possible shape mismatch.")
                    raise e
                
                self.data.append((padded_video, gloss))
    
    def _pad_video(self, video, max_frames):
        processed_frames = []
        for frame in video:
            if len(frame) != 49:
                raise ValueError(f"Expected 49 joints per frame, but got {len(frame)}.")
            
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
        video, gloss = self.data[idx]
        video_tensor = torch.tensor(video)

        cond_vector = self.glove_embeddings.get(gloss, np.zeros(self.embedding_dim))
        cond_vector = torch.tensor(cond_vector, dtype=torch.float32)
        
        sample = {'video': video_tensor, 'condition': cond_vector}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, outputs, mask):
        scores = self.attn(outputs).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)
        return context, attn_weights

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_prob=0.2):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        self.fc_mu = nn.Linear(hidden_dim * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 4, latent_dim)

        # Define BatchNorm layers correctly
        self.batch_norm_hidden = nn.BatchNorm1d(hidden_dim * 2)  # For LSTM outputs
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        B, T, _ = x.size()

        mask = (x.sum(dim=2) != 0).float()
        outputs, (h_n, _) = self.lstm(x)

        # Apply batch norm to LSTM outputs
        outputs = outputs.permute(0, 2, 1)  # (batch_size, hidden_dim * 2, seq_length)
        outputs = self.batch_norm_hidden(outputs)  # Apply BatchNorm
        outputs = outputs.permute(0, 2, 1)  # Back to (batch_size, seq_length, hidden_dim * 2)

        # Applying dropout
        outputs = self.dropout(outputs)

        # Attention mechanism
        context, attn_weights = self.attention(outputs, mask)

        mu = self.fc_mu(torch.cat([outputs.mean(dim=1), context], dim=-1))
        logvar = self.fc_logvar(torch.cat([outputs.mean(dim=1), context], dim=-1))


        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        return z, mu, logvar, attn_weights


class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim, output_dim, seq_len, dropout_prob=0.2):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim + cond_dim, latent_dim + cond_dim)
        self.lstm = nn.LSTM(latent_dim + cond_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # Batch Normalization after LSTM output
        self.dropout = nn.Dropout(p=dropout_prob)  # Adding dropout to Decoder
    
    def forward(self, z, cond):
        B = z.size(0)
        z_cond = torch.cat([z, cond], dim=-1)
        z_cond = torch.relu(self.fc(z_cond))
        
        z_seq = z_cond.unsqueeze(1).repeat(1, self.seq_len, 1)
        outputs, _ = self.lstm(z_seq)
        
        # Applying dropout
        z_cond = self.dropout(z_cond)
        
        # Reshape outputs for BatchNorm (batch_size, features, sequence_length)
        outputs = outputs.permute(0, 2, 1)  # Change to (batch_size, hidden_dim, sequence_length)
        outputs = self.batch_norm(outputs)  # Apply BatchNorm
        outputs = outputs.permute(0, 2, 1)  # Change back to (batch_size, sequence_length, hidden_dim)
        
        # Applying dropout before the final output layer
        outputs = self.dropout(outputs)
        
        recon_seq = torch.tanh(self.fc_out(outputs))
        return recon_seq

class LatentClassifier(nn.Module):
    def __init__(self, latent_dim):
        super(LatentClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  
        )
    
    def forward(self, z):
        return self.fc(z)

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dim, output_dim, seq_len):
        super(ConditionalVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, cond_dim, hidden_dim, output_dim, seq_len)
        self.latent_classifier = LatentClassifier(latent_dim)  
    
    def forward(self, x, cond):
        z, mu, logvar, attn_weights = self.encoder(x)
        recon_x = self.decoder(z, cond)
        return recon_x, mu, logvar, attn_weights, z

def kl_divergence_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def latent_classification_loss(z_v, z_g, latent_classifier):
    pred_z_v = latent_classifier(z_v)
    pred_z_g = latent_classifier(z_g)
    
    loss_z_v = nn.functional.binary_cross_entropy(pred_z_v, torch.ones_like(pred_z_v))
    loss_z_g = nn.functional.binary_cross_entropy(pred_z_g, torch.zeros_like(pred_z_g))
    
    return loss_z_v + loss_z_g

def train(model, train_loader, val_loader, device, num_epochs, lr):
    print("Starting training...")
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}...")
        epoch_loss = 0.0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            video = batch['video'].to(device)
            condition = batch['condition'].to(device)

            with autocast():
                recon_video, mu, logvar, attn_weights, z_v = model(video, condition)
                z_g = torch.normal(0, 1, size=z_v.shape).to(device)
                loss = kl_divergence_loss(mu, logvar) + latent_classification_loss(z_v, z_g, model.latent_classifier)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"  Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(device)
                condition = batch['condition'].to(device)
                recon_video, mu, logvar, attn_weights, z_v = model(video, condition)
                z_g = torch.normal(0, 1, size=z_v.shape).to(device)
                loss = kl_divergence_loss(mu, logvar) + latent_classification_loss(z_v, z_g, model.latent_classifier)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1} completed. Avg Loss: {epoch_loss/len(train_loader.dataset):.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


SEQ_LEN = 233 
INPUT_DIM = 147  
HIDDEN_DIM = 128
LATENT_DIM = 64
BATCH_SIZE = 16
MERGED_JSONL_PATH = "Dataset/0_landmarks.jsonl"

print("\nInitializing dataset...")
dataset = LandmarkDataset(MERGED_JSONL_PATH)

SEQ_LEN = dataset.max_frames
print(f"Max frame count (sequence length): {SEQ_LEN}")
print("Vocabulary:", dataset.vocab)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

cond_dim = dataset.embedding_dim  

print("\nInitializing model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalVAE(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM,
    cond_dim=cond_dim,
    output_dim=INPUT_DIM,
    seq_len=SEQ_LEN
).to(device)

print("\nModel initialized. Starting training...\n")
train(model, dataloader, device, num_epochs=100, lr=0.001)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'latent_classifier_state_dict': model.latent_classifier.state_dict()
}, "cvae_model.pth")

print("Model saved successfully as 'cvae_model.pth'")

