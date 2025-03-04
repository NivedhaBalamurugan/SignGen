import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
import numpy as np
import logging
import gc

class AttentionModule(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4, 
            dropout=0.2, 
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        attn_output, attn_weights = self.attention(x, x, x, key_padding_mask=mask)
        return self.layer_norm(x + attn_output), attn_weights

class EnhancedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len, num_joints, dropout_prob=0.3):
        super(EnhancedEncoder, self).__init__()
        
        # Input projection and processing
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Bidirectional LSTM with residual connections
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout_prob
        )
        
        # Attention module
        self.attention = AttentionModule(hidden_dim * 2)
        
        # Latent space mapping with more stability
        self.latent_mu = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, latent_dim)
        )
        
        self.latent_logvar = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, latent_dim)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
        # Store dimensions
        self.seq_len = seq_len
        self.num_joints = num_joints

    def forward(self, x):
        # Ensure correct input shape: (Batch, Seq_Len, Num_Joints, Coords)
        B, T, J, C = x.size()
        
        # Reshape to (Batch, Seq_Len, Num_Joints * Coords)
        x = x.view(B, T, -1)
        
        # Frame filtering logic
        x_joints = x.view(B, T, self.num_joints, 3)
        
        # Validate frames: check for non-zero joints
        frame_not_padded = (x_joints.abs().sum(dim=(2, 3)) > 0).float()
        
        # Input projection
        x_masked = x * frame_not_padded.unsqueeze(-1)
        x_proj = self.input_proj(x_masked)
        
        # Pack padded sequence
        lengths = frame_not_padded.sum(dim=1).int().cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x_proj, 
            lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # LSTM processing
        packed_outputs, _ = self.lstm(packed_input)
        
        # Unpack the sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, 
            batch_first=True, 
            total_length=T
        )
        
        # Attention
        outputs, attn_weights = self.attention(outputs)
        
        # Mask attention outputs
        outputs = outputs * frame_not_padded.unsqueeze(-1)
        
        # Global pooling considering only valid frames
        context = (outputs * frame_not_padded.unsqueeze(-1)).mean(dim=1)
        context = self.dropout(context)
        
        # Latent space computation
        mu = self.latent_mu(torch.cat([outputs.mean(dim=1), context], dim=-1))
        logvar = self.latent_logvar(torch.cat([outputs.mean(dim=1), context], dim=-1))
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        
        return z, mu, logvar, attn_weights

class EnhancedDecoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim, output_dim, seq_len, num_joints):
        super(EnhancedDecoder, self).__init__()
        # Condition and latent fusion
        self.condition_fusion = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # More complex LSTM decoder with residual connections
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.3
        )
        
        # Multi-layer output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()  # Constrain output
        )
        
        self.seq_len = seq_len
        self.num_joints = num_joints
        self.output_dim = output_dim

    def forward(self, z, cond):
        # Fuse latent space and condition
        z_cond = torch.cat([z, cond], dim=-1)
        z_cond = self.condition_fusion(z_cond)
        
        # Repeat for sequence length
        z_seq = z_cond.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # LSTM processing
        outputs, _ = self.lstm(z_seq)
        
        # Reconstruct output
        recon_seq = self.output_proj(outputs)
        
        # Reshape to original input shape (Batch, Seq_Len, Num_Joints, Coords)
        recon_seq = recon_seq.view(-1, self.seq_len, self.num_joints, self.output_dim // self.num_joints)
        
        return recon_seq

class LatentClassifier(nn.Module):
    def __init__(self, latent_dim, dropout_prob=0.3):
        super(LatentClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.classifier(z)

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dim, output_dim, seq_len, num_joints=49):
        super(ConditionalVAE, self).__init__()
        self.encoder = EnhancedEncoder(
            input_dim, 
            hidden_dim, 
            latent_dim, 
            seq_len, 
            num_joints
        )
        self.decoder = EnhancedDecoder(
            latent_dim, 
            cond_dim, 
            hidden_dim, 
            output_dim, 
            seq_len, 
            num_joints
        )
        self.latent_classifier = LatentClassifier(latent_dim)

    def forward(self, x, cond):
        z, mu, logvar, attn_weights = self.encoder(x)
        recon_x = self.decoder(z, cond)
        return recon_x, mu, logvar, attn_weights, z

def improved_kl_divergence_loss(mu, logvar, beta=1.0):
    """
    More stable KL divergence with adaptive weight
    """
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return beta * KLD

def adaptive_reconstruction_loss(recon_x, x):
    """
    Combine multiple loss metrics for better reconstruction
    """
    # Ensure tensors have same shape before loss computation
    assert recon_x.shape == x.shape, f"Shape mismatch: recon_x {recon_x.shape}, x {x.shape}"
    
    # Smooth L1 Loss
    smooth_l1 = F.smooth_l1_loss(recon_x, x, reduction='mean')
    
    # MSE Loss
    mse = F.mse_loss(recon_x, x, reduction='mean')
    
    # Huber Loss
    huber = F.huber_loss(recon_x, x, reduction='mean')
    
    # Weighted combination
    return 0.4 * smooth_l1 + 0.3 * mse + 0.3 * huber

def adaptive_latent_classification_loss(z_v, z_g, latent_classifier, margin=1.0):
    """
    More stable latent classification loss with margin
    """
    pred_z_v = latent_classifier(z_v)
    pred_z_g = latent_classifier(z_g)
    
    # Margin-based loss
    loss_z_v = torch.mean(F.binary_cross_entropy(pred_z_v, torch.ones_like(pred_z_v)) * margin)
    loss_z_g = torch.mean(F.binary_cross_entropy(pred_z_g, torch.zeros_like(pred_z_g)) * margin)
    
    return loss_z_v + loss_z_g