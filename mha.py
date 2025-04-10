import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *  
import new_cvae_inf
import numpy as np
import show_output
import cgan_inference
import os

class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):  # Changed to 4 heads to be divisible by 58
        super(MultiHeadAttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Ensure feature_dim is divisible by num_heads
        assert feature_dim % num_heads == 0, f"Feature dimension ({feature_dim}) must be divisible by number of heads ({num_heads})"
        self.head_dim = feature_dim // num_heads
        
        # Multi-head attention layers
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(feature_dim, feature_dim)
        
        # Fusion weights (learnable parameters)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, gan_seq, vae_seq):
        # gan_seq, vae_seq shape: [batch_size, seq_len, feature_dim]
        batch_size, seq_len, _ = gan_seq.shape
        
        # Use GAN sequence as query, VAE sequence for keys and values
        q = self.query(gan_seq).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(vae_seq).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(vae_seq).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.feature_dim)
        
        # Final output with residual connection
        output = self.output_layer(context)
        
        # Adaptive fusion with learnable parameter
        fused_seq = self.alpha * gan_seq + (1 - self.alpha) * output
        
        return fused_seq

class SignSequenceFuser(nn.Module):
    def __init__(self, feature_dim, num_layers=2, num_heads=2):  # Added num_heads parameter
        super(SignSequenceFuser, self).__init__()
        self.layers = nn.ModuleList([
            MultiHeadAttentionFusion(feature_dim, num_heads=num_heads) for _ in range(num_layers)
        ])
        
        # Final feature refinement
        self.feature_refinement = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, gan_seq, vae_seq):
        # Initial fusion
        fused_seq = self.layers[0](gan_seq, vae_seq)
        
        # Refine through additional layers
        for layer in self.layers[1:]:
            fused_seq = layer(fused_seq, vae_seq)
            
        # Final refinement
        output = self.feature_refinement(fused_seq)
        
        # Add residual connection to preserve sequence integrity
        final_output = output + 0.5 * (gan_seq + vae_seq)
        
        return final_output


def fuse_sequences(input_word):
    gan_sequence = new_cvae_inf.get_cvae_sequence(input_word)  
    vae_sequence = cgan_inference.get_cgan_sequence(input_word)
    
    gan_tensor = torch.FloatTensor(gan_sequence) 
    vae_tensor = torch.FloatTensor(vae_sequence)  
    
    gan_flat = gan_tensor.reshape(30, -1)  
    vae_flat = vae_tensor.reshape(30, -1)  

    similarity = torch.matmul(gan_flat, vae_flat.transpose(0, 1))  
    attention_weights = torch.softmax(similarity, dim=1)  
    
    weighted_vae = torch.matmul(attention_weights, vae_flat).reshape(30, 29, 2)
    
    alpha = 0.6 
    fused_tensor = alpha * gan_tensor + (1 - alpha) * weighted_vae
    
    fused_sequence = fused_tensor.numpy()
    
    show_output.save_generated_sequence(fused_sequence, MHA_OUTPUT_FRAMES, MHA_OUTPUT_VIDEO)
    return fused_sequence

if __name__ == "__main__":
    fuse_sequences("hat")