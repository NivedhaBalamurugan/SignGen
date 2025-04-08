import os
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
import orjson
import gc
import psutil
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
import torch.cuda.amp as amp
import warnings
from tqdm import tqdm
from functools import lru_cache
from config import *


# # class TemporalAttention(nn.Module):
# #     def __init__(self, hidden_size):
# #         super(TemporalAttention, self).__init__()
# #         self.hidden_size = hidden_size
# #         self.attn = nn.Linear(hidden_size * 2, hidden_size)
# #         self.v = nn.Parameter(torch.rand(hidden_size))
# #         self.v.data.normal_(mean=0, std=1/np.sqrt(self.v.size(0)))

# #     def forward(self, hidden, encoder_outputs):
# #         # hidden: [batch, hidden_size]
# #         # encoder_outputs: [batch, seq_len, hidden_size]

# #         seq_len = encoder_outputs.size(1)
# #         batch_size = encoder_outputs.size(0)

# #         # Repeat hidden for each timestep
# #         hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

# #         # Concatenate hidden and encoder outputs
# #         energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

# #         # Calculate attention weights
# #         energy = energy.permute(0, 2, 1)  # [batch, hidden, seq_len]
# #         v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch, 1, hidden]
# #         attention = torch.bmm(v, energy)  # [batch, 1, seq_len]
# #         attention = F.softmax(attention.squeeze(1), dim=1)  # [batch, seq_len]

# #         # Apply attention to encoder outputs
# #         context = torch.bmm(attention.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden]
# #         context = context.squeeze(1)  # [batch, hidden]

# #         return context, attention

# # class JointAttention(nn.Module):
# #     def __init__(self, num_joints, hidden_dim):
# #         super(JointAttention, self).__init__()
# #         self.joint_query = nn.Linear(hidden_dim, hidden_dim)
# #         self.joint_key = nn.Linear(hidden_dim, hidden_dim)
# #         self.joint_value = nn.Linear(hidden_dim, hidden_dim)
# #         self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

# #     def forward(self, joint_features):
# #         # Move scale to the same device as inputs
# #         self.scale = self.scale.to(joint_features.device)

# #         # joint_features: [batch, num_joints, hidden_dim]

# #         # Generate queries, keys, and values
# #         Q = self.joint_query(joint_features)  # [batch, num_joints, hidden_dim]
# #         K = self.joint_key(joint_features)    # [batch, num_joints, hidden_dim]
# #         V = self.joint_value(joint_features)  # [batch, num_joints, hidden_dim]

# #         # Calculate attention scores
# #         attention = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale  # [batch, num_joints, num_joints]

# #         # Apply softmax to get attention weights
# #         attention_weights = F.softmax(attention, dim=2)

# #         # Apply attention weights to values
# #         context = torch.matmul(attention_weights, V)  # [batch, num_joints, hidden_dim]

# #         return context, attention_weights

# # class CVAE(nn.Module):
# #      def __init__(self, num_frames=30, num_line_seg=27, num_joints=2, latent_size=64, embedding_dim=EMBEDDING_DIM):
# #         super(CVAE, self).__init__()
# #         self.num_frames = num_frames
# #         self.num_line_seg = num_line_seg
# #         self.num_joints = num_joints
# #         self.latent_size = latent_size
# #         self.embedding_dim = embedding_dim
# #         self.coord_dim = 2  # x,y coordinates
# #         self.hidden_size = 128

# #         # Encoder - Temporal modeling approach
# #         self.encoder_lstm = nn.LSTM(
# #         input_size=num_line_seg * num_joints * self.coord_dim,  # 27 * 2 * 2 = 108
# #         hidden_size=self.hidden_size,
# #         num_layers=2,
# #         batch_first=True,
# #         bidirectional=True
# #        )

# #         # Decoder LSTM to generate temporal data
# #         self.decoder_lstm = nn.LSTM(
# #             input_size=self.hidden_size,
# #             hidden_size=self.hidden_size,
# #             num_layers=2,
# #             batch_first=True
# #         )

# #         # Temporal attention for encoder
# #         self.temporal_attention = TemporalAttention(self.hidden_size * 2)  # *2 for bidirectional

# #         # Joint attention for spatial relationships
# #         self.joint_attention = JointAttention(num_joints, 64)

# #         # Fully connected layers for encoding
# #         self.fc_encoder = nn.Sequential(
# #             nn.Linear(self.hidden_size * 2 + embedding_dim, 512),
# #             nn.ReLU(),
# #             nn.Linear(512, 256),
# #             nn.ReLU()
# #         )

# #         # Latent space
# #         self.mu = nn.Linear(256, self.latent_size)
# #         self.logvar = nn.Linear(256, self.latent_size)

# #         # Decoder FC layers
# #         self.fc_decoder = nn.Sequential(
# #             nn.Linear(self.latent_size + embedding_dim, 256),
# #             nn.ReLU(),
# #             nn.Linear(256, 512),
# #             nn.ReLU(),
# #             nn.Linear(512, self.hidden_size),
# #             nn.ReLU()
# #         )

# #         # Attention for decoder
# #         self.decoder_attention = TemporalAttention(self.hidden_size)

# #         # Joint feature projection
# #         self.joint_projection = nn.Linear(self.hidden_size, num_joints * 64)

# #         # Final projection to joint coordinates
# #         self.output_fc = nn.Linear(64, self.coord_dim)


# #      def encode(self, x, condition):
# #         # x shape: [batch_size, num_frames, num_line_seg, num_joints, 2]
# #         batch_size = x.shape[0]

# #         # Reshape for LSTM: [batch, seq_len, features]
# #         x_reshaped = x.view(batch_size, self.num_frames, -1)

# #         # Process sequence with LSTM
# #         lstm_out, (h_n, _) = self.encoder_lstm(x_reshaped)

# #         # Get final hidden state
# #         h_n = h_n[-2:].transpose(0, 1).contiguous().view(batch_size, -1)  # Combine bidirectional

# #         # Apply temporal attention
# #         context, _ = self.temporal_attention(h_n, lstm_out)

# #         sample_frame_idx = self.num_frames // 2
# #         sample_line_idx = self.num_line_seg // 2  # Just take a middle line segment

# #         if len(x.shape) < 5:
# #             x_full = x.view(batch_size, self.num_frames, self.num_line_seg, self.num_joints, 2)
# #             joint_input = x_full[:, sample_frame_idx, sample_line_idx]
# #         else:
# #             joint_input = x[:, sample_frame_idx, sample_line_idx]

# #         # Project to joint feature space for attention
# #         joint_features = nn.Linear(2, 64).to(x.device)(joint_input)

# #         # Apply joint attention for spatial relationships
# #         joint_context, _ = self.joint_attention(joint_features)

# #         # Flatten and project joint context
# #         joint_context = joint_context.reshape(batch_size, -1)
# #         joint_context = nn.Linear(self.num_joints * 64, self.hidden_size).to(x.device)(joint_context)

# #         # Concatenate temporal context, joint context and condition information
# #         combined = torch.cat([context, joint_context, condition], dim=1)

# #         # Adjust encoder to handle additional input
# #         h = nn.Linear(self.hidden_size * 3 + self.embedding_dim, 512).to(x.device)(combined)
# #         h = F.relu(h)
# #         h = nn.Linear(512, 256).to(x.device)(h)
# #         h = F.relu(h)

# #         # Get latent parameters
# #         mu = self.mu(h)
# #         logvar = self.logvar(h)

# #         return mu, logvar

# #      def reparameterize(self, mu, logvar):
# #         std = torch.exp(0.5 * logvar)
# #         # Use the device of mu for eps
# #         eps = torch.randn_like(std)
# #         return eps * std + mu

# #      def decode(self, z):
# #         batch_size = z.shape[0]

# #         # Process through FC layers
# #         h = self.fc_decoder(z)

# #         # Initialize decoder hidden state
# #         h_0 = torch.zeros(2, batch_size, self.hidden_size).to(z.device)
# #         c_0 = torch.zeros(2, batch_size, self.hidden_size).to(z.device)

# #         # Initialize first input
# #         decoder_input = h.unsqueeze(1).repeat(1, self.num_frames, 1)

# #         # Process with decoder LSTM
# #         lstm_out, _ = self.decoder_lstm(decoder_input, (h_0, c_0))

# #         # Create output tensor with correct dimensions
# #         # [batch_size, num_frames, num_line_seg, num_joints, 2]
# #         output = torch.zeros(batch_size, self.num_frames, self.num_line_seg, self.num_joints, self.coord_dim).to(z.device)

# #         # Apply attention over time for each frame
# #         for t in range(self.num_frames):
# #             # Get the hidden state for this frame
# #             frame_hidden = lstm_out[:, t]

# #             # Apply temporal attention
# #             context, _ = self.decoder_attention(frame_hidden, lstm_out)

# #             # For each line segment
# #             for l in range(self.num_line_seg):
# #                 # Project to joint features - ensure we create features per line segment
# #                 # We'll use the same context but project it differently for each line segment
# #                 line_context = context + l * 0.01  # Small offset to differentiate line segments
# #                 joint_features = self.joint_projection(line_context).view(batch_size, self.num_joints, 64)

# #                 # Apply joint attention for spatial relationships
# #                 attended_joints, _ = self.joint_attention(joint_features)

# #                 # Project to coordinates
# #                 coords = self.output_fc(attended_joints)  # [batch_size, num_joints, 2]

# #                 # Store in the output tensor
# #                 output[:, t, l] = coords

# #         return output

# #      def forward(self, x, condition):
# #         x = x.to(torch.float32)
# #         condition = condition.to(torch.float32)
# #         # Encode
# #         mu, logvar = self.encode(x, condition)

# #         # Sample latent vector
# #         z = self.reparameterize(mu, logvar)

# #         # Add condition to latent
# #         z_conditioned = torch.cat([z.to(torch.float32), condition.to(torch.float32)], dim=1)

# #         # Decode
# #         recon = self.decode(z_conditioned)

# #         # Return the input x along with other values for loss calculation
# #         return recon, mu, logvar, x

# # def loss_function(x, recon, mu, logvar, beta=1.0):
# #     # Reconstruction loss - MSE over all coordinates
# #     recon_loss = F.mse_loss(recon, x, reduction='sum')

# #     # KL divergence
# #     kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# #     # Total loss
# #     return recon_loss, beta * kld

# # def beta_schedule(epoch, total_epochs, beta_min=0.01, beta_max=1.0, warmup_fraction=0.1):
# #     warmup_epochs = int(total_epochs * warmup_fraction)
# #     if epoch < warmup_epochs:
# #         # Linear warmup
# #         return beta_min + (beta_max - beta_min) * (epoch / warmup_epochs)
# #     else:
# #         # Cosine annealing after warmup
# #         return beta_min + 0.5 * (beta_max - beta_min) * (1 - np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))





# class TemporalAttention(nn.Module):
#     def __init__(self, hidden_size):
#         super(TemporalAttention, self).__init__()
#         self.hidden_size = hidden_size
#         self.attn = nn.Linear(hidden_size * 2, hidden_size)
#         self.v = nn.Parameter(torch.rand(hidden_size))
#         self.v.data.normal_(mean=0, std=1/np.sqrt(self.v.size(0)))

#     def forward(self, hidden, encoder_outputs):
#         # hidden: [batch, hidden_size]
#         # encoder_outputs: [batch, seq_len, hidden_size]

#         seq_len = encoder_outputs.size(1)
#         batch_size = encoder_outputs.size(0)

#         # Repeat hidden for each timestep
#         hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

#         # Concatenate hidden and encoder outputs
#         energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

#         # Calculate attention weights
#         energy = energy.permute(0, 2, 1)  # [batch, hidden, seq_len]
#         v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch, 1, hidden]
#         attention = torch.bmm(v, energy)  # [batch, 1, seq_len]
#         attention = F.softmax(attention.squeeze(1), dim=1)  # [batch, seq_len]

#         # Apply attention to encoder outputs
#         context = torch.bmm(attention.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden]
#         context = context.squeeze(1)  # [batch, hidden]

#         return context, attention

# class JointAttention(nn.Module):
#     def __init__(self, num_joints, hidden_dim):
#         super(JointAttention, self).__init__()
#         self.joint_query = nn.Linear(hidden_dim, hidden_dim)
#         self.joint_key = nn.Linear(hidden_dim, hidden_dim)
#         self.joint_value = nn.Linear(hidden_dim, hidden_dim)
#         self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

#     def forward(self, joint_features):
#         # Move scale to the same device as inputs
#         self.scale = self.scale.to(joint_features.device)

#         # joint_features: [batch, num_joints, hidden_dim]

#         # Generate queries, keys, and values
#         Q = self.joint_query(joint_features)  # [batch, num_joints, hidden_dim]
#         K = self.joint_key(joint_features)    # [batch, num_joints, hidden_dim]
#         V = self.joint_value(joint_features)  # [batch, num_joints, hidden_dim]

#         # Calculate attention scores
#         attention = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale  # [batch, num_joints, num_joints]

#         # Apply softmax to get attention weights
#         attention_weights = F.softmax(attention, dim=2)

#         # Apply attention weights to values
#         context = torch.matmul(attention_weights, V)  # [batch, num_joints, hidden_dim]

#         return context, attention_weights

# class CVAE(nn.Module):
#     def __init__(self, num_frames=30, num_joints=29, latent_size=64, embedding_dim=EMBEDDING_DIM):
#         super(CVAE, self).__init__()
#         self.num_frames = num_frames
#         self.num_joints = num_joints
#         self.latent_size = latent_size
#         self.embedding_dim = embedding_dim
#         self.coord_dim = 2  # x,y coordinates
#         self.hidden_size = 128

#         # Encoder - Temporal modeling approach
#         self.encoder_lstm = nn.LSTM(
#             input_size=num_joints * self.coord_dim,  # 29 * 2 = 58
#             hidden_size=self.hidden_size,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True
#         )

#         # Decoder LSTM to generate temporal data
#         self.decoder_lstm = nn.LSTM(
#             input_size=self.hidden_size,
#             hidden_size=self.hidden_size,
#             num_layers=2,
#             batch_first=True
#         )

#         # Temporal attention for encoder
#         self.temporal_attention = TemporalAttention(self.hidden_size * 2)  # *2 for bidirectional

#         # Joint attention for spatial relationships
#         self.joint_attention = JointAttention(num_joints, 64)

#         # Fully connected layers for encoding
#         self.fc_encoder = nn.Sequential(
#             nn.Linear(self.hidden_size * 2 + embedding_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU()
#         )

#         # Latent space
#         self.mu = nn.Linear(256, self.latent_size)
#         self.logvar = nn.Linear(256, self.latent_size)

#         # Decoder FC layers
#         self.fc_decoder = nn.Sequential(
#             nn.Linear(self.latent_size + embedding_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, self.hidden_size),
#             nn.ReLU()
#         )

#         # Attention for decoder
#         self.decoder_attention = TemporalAttention(self.hidden_size)

#         # Joint feature projection
#         self.joint_projection = nn.Linear(self.hidden_size, num_joints * 64)

#         # Final projection to joint coordinates
#         self.output_fc = nn.Linear(64, self.coord_dim)

#     def encode(self, x, condition):
#         # x shape: [batch_size, num_frames, num_joints, 2]
#         batch_size = x.shape[0]

#         # Reshape for LSTM: [batch, seq_len, features]
#         x_reshaped = x.view(batch_size, self.num_frames, -1)

#         # Process sequence with LSTM
#         lstm_out, (h_n, _) = self.encoder_lstm(x_reshaped)

#         # Get final hidden state
#         h_n = h_n[-2:].transpose(0, 1).contiguous().view(batch_size, -1)  # Combine bidirectional

#         # Apply temporal attention
#         context, _ = self.temporal_attention(h_n, lstm_out)

#         # Get joint features from middle frame
#         sample_frame_idx = self.num_frames // 2
#         joint_input = x[:, sample_frame_idx]

#         # Project to joint feature space for attention
#         joint_features = nn.Linear(2, 64).to(x.device)(joint_input)

#         # Apply joint attention for spatial relationships
#         joint_context, _ = self.joint_attention(joint_features)

#         # Flatten and project joint context
#         joint_context = joint_context.reshape(batch_size, -1)
#         joint_context = nn.Linear(self.num_joints * 64, self.hidden_size).to(x.device)(joint_context)

#         # Concatenate temporal context, joint context and condition information
#         combined = torch.cat([context, joint_context, condition], dim=1)

#         # Adjust encoder to handle additional input
#         h = nn.Linear(self.hidden_size * 3 + self.embedding_dim, 512).to(x.device)(combined)
#         h = F.relu(h)
#         h = nn.Linear(512, 256).to(x.device)(h)
#         h = F.relu(h)

#         # Get latent parameters
#         mu = self.mu(h)
#         logvar = self.logvar(h)

#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu

#     def decode(self, z):
#         batch_size = z.shape[0]

#         # Process through FC layers
#         h = self.fc_decoder(z)

#         # Initialize decoder hidden state
#         h_0 = torch.zeros(2, batch_size, self.hidden_size).to(z.device)
#         c_0 = torch.zeros(2, batch_size, self.hidden_size).to(z.device)

#         # Initialize first input
#         decoder_input = h.unsqueeze(1).repeat(1, self.num_frames, 1)

#         # Process with decoder LSTM
#         lstm_out, _ = self.decoder_lstm(decoder_input, (h_0, c_0))

#         # Create output tensor with correct dimensions
#         # [batch_size, num_frames, num_joints, 2]
#         output = torch.zeros(batch_size, self.num_frames, self.num_joints, self.coord_dim).to(z.device)

#         # Apply attention over time for each frame
#         for t in range(self.num_frames):
#             # Get the hidden state for this frame
#             frame_hidden = lstm_out[:, t]

#             # Apply temporal attention
#             context, _ = self.decoder_attention(frame_hidden, lstm_out)

#             # Project to joint features
#             joint_features = self.joint_projection(context).view(batch_size, self.num_joints, 64)

#             # Apply joint attention for spatial relationships
#             attended_joints, _ = self.joint_attention(joint_features)

#             # Project to coordinates
#             coords = self.output_fc(attended_joints)  # [batch_size, num_joints, 2]

#             # Store in the output tensor
#             output[:, t] = coords

#         return output

#     def forward(self, x, condition):
#         x = x.to(torch.float32)
#         condition = condition.to(torch.float32)
#         # Encode
#         mu, logvar = self.encode(x, condition)

#         # Sample latent vector
#         z = self.reparameterize(mu, logvar)

#         # Add condition to latent
#         z_conditioned = torch.cat([z.to(torch.float32), condition.to(torch.float32)], dim=1)

#         # Decode
#         recon = self.decode(z_conditioned)

#         # Return the input x along with other values for loss calculation
#         return recon, mu, logvar, x

# def loss_function(x, recon, mu, logvar, beta=1.0):
#     # Reconstruction loss - MSE over all coordinates
#     recon_loss = F.mse_loss(recon, x, reduction='sum')

#     # KL divergence
#     kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     # Total loss
#     return recon_loss, beta * kld

# def beta_schedule(epoch, total_epochs, beta_min=0.01, beta_max=1.0, warmup_fraction=0.1):
#     warmup_epochs = int(total_epochs * warmup_fraction)
#     if epoch < warmup_epochs:
#         # Linear warmup
#         return beta_min + (beta_max - beta_min) * (epoch / warmup_epochs)
#     else:
#         # Cosine annealing after warmup
#         return beta_min + 0.5 * (beta_max - beta_min) * (1 - np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))



# class TemporalAttention(nn.Module):
#     def __init__(self, hidden_size):
#         super(TemporalAttention, self).__init__()
#         self.hidden_size = hidden_size
#         self.attn = nn.Linear(hidden_size * 2, hidden_size)
#         self.v = nn.Parameter(torch.rand(hidden_size))
#         self.v.data.normal_(mean=0, std=1/np.sqrt(self.v.size(0)))

#     def forward(self, hidden, encoder_outputs):
#         # hidden: [batch, hidden_size]
#         # encoder_outputs: [batch, seq_len, hidden_size]

#         seq_len = encoder_outputs.size(1)
#         batch_size = encoder_outputs.size(0)

#         # Repeat hidden for each timestep
#         hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

#         # Concatenate hidden and encoder outputs
#         energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

#         # Calculate attention weights
#         energy = energy.permute(0, 2, 1)  # [batch, hidden, seq_len]
#         v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch, 1, hidden]
#         attention = torch.bmm(v, energy)  # [batch, 1, seq_len]
#         attention = F.softmax(attention.squeeze(1), dim=1)  # [batch, seq_len]

#         # Apply attention to encoder outputs
#         context = torch.bmm(attention.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden]
#         context = context.squeeze(1)  # [batch, hidden]

#         return context, attention


# # Modified JointAttention class with zero-joint handling
# class JointAttention(nn.Module):
#     def __init__(self, num_joints, hidden_dim):
#         super(JointAttention, self).__init__()
#         self.joint_query = nn.Linear(hidden_dim, hidden_dim)
#         self.joint_key = nn.Linear(hidden_dim, hidden_dim)
#         self.joint_value = nn.Linear(hidden_dim, hidden_dim)
#         self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

#     def forward(self, joint_features, joint_mask=None):
#         # Move scale to the same device as inputs
#         self.scale = self.scale.to(joint_features.device)

#         # joint_features: [batch, num_joints, hidden_dim]
#         # joint_mask: [batch, num_joints] (1 if valid, 0 if coordinates were zero)

#         # Generate queries, keys, and values
#         Q = self.joint_query(joint_features)  # [batch, num_joints, hidden_dim]
#         K = self.joint_key(joint_features)    # [batch, num_joints, hidden_dim]
#         V = self.joint_value(joint_features)  # [batch, num_joints, hidden_dim]

#         # Calculate attention scores
#         attention = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale  # [batch, num_joints, num_joints]

#         # Apply mask if provided (set attention scores to -inf for zero joints)
#         if joint_mask is not None:
#             # Create attention mask (set scores to -inf where joints are zero)
#             # Expand mask to match attention dimensions
#             mask = joint_mask.unsqueeze(1).expand(-1, attention.size(1), -1)
#             attention = attention.masked_fill(mask == 0, -1e9)

#         # Apply softmax to get attention weights
#         attention_weights = F.softmax(attention, dim=2)

#         # If we have mask, zero out attention to invalid joints
#         if joint_mask is not None:
#             attention_weights = attention_weights * mask

#         # Apply attention weights to values
#         context = torch.matmul(attention_weights, V)  # [batch, num_joints, hidden_dim]

#         return context, attention_weights


# # Modified CVAE class with improvements
# class CVAE(nn.Module):
#     def __init__(self, num_frames=30, num_joints=29, latent_size=64, embedding_dim=EMBEDDING_DIM, kl_weight=0.1):
#         super(CVAE, self).__init__()
#         self.num_frames = num_frames
#         self.num_joints = num_joints
#         self.latent_size = latent_size
#         self.embedding_dim = embedding_dim
#         self.coord_dim = 2  # x,y coordinates
#         self.hidden_size = 128
#         self.kl_weight = kl_weight

#         # Encoder - Temporal modeling approach
#         self.encoder_lstm = nn.LSTM(
#             input_size=num_joints * self.coord_dim,  # 29 * 2 = 58
#             hidden_size=self.hidden_size,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True
#         )

#         # Condition projection layer for encoder
#         self.condition_encoder = nn.Sequential(
#             nn.Linear(embedding_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64)
#         )

#         # Condition projection layer for decoder
#         self.condition_decoder = nn.Sequential(
#             nn.Linear(embedding_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64)
#         )

#         # Decoder LSTM to generate temporal data
#         self.decoder_lstm = nn.LSTM(
#             input_size=self.hidden_size + 64,  # Added condition dimension
#             hidden_size=self.hidden_size,
#             num_layers=2,
#             batch_first=True
#         )

#         # Temporal attention for encoder
#         self.temporal_attention = TemporalAttention(self.hidden_size * 2)  # *2 for bidirectional

#         # Joint attention for spatial relationships
#         self.joint_attention = JointAttention(num_joints, 64)

#         # Joint feature extractor
#         self.joint_feature_extractor = nn.Linear(2, 64)

#         # Fully connected layers for encoding
#         self.fc_encoder = nn.Sequential(
#             nn.Linear(self.hidden_size * 2 + self.hidden_size + 64, 512),  # Condition added
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU()
#         )

#         # Latent space
#         self.mu = nn.Linear(256, self.latent_size)
#         self.logvar = nn.Linear(256, self.latent_size)

#         # Decoder FC layers
#         self.fc_decoder = nn.Sequential(
#             nn.Linear(self.latent_size + 64, 256),  # Changed to use processed condition
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, self.hidden_size),
#             nn.ReLU()
#         )

#         # Final projection layers
#         self.joint_projection = nn.Linear(self.hidden_size, num_joints * 64)
#         self.output_fc = nn.Linear(64, self.coord_dim)

#     def create_joint_mask(self, x):
#         # Create mask where both x and y coordinates are non-zero
#         # x shape: [batch_size, num_frames, num_joints, 2]
#         # Output: [batch_size, num_frames, num_joints]
        
#         # Check if both x and y coordinates are zero
#         joint_mask = (x.abs().sum(dim=-1) > 0).float()
#         return joint_mask

#     def encode(self, x, condition):
#         # x shape: [batch_size, num_frames, num_joints, 2]
#         batch_size = x.shape[0]

#         # Process condition for encoder
#         condition_enc = self.condition_encoder(condition)  # [batch, 64]

#         # Create mask for zero joints
#         joint_mask = self.create_joint_mask(x)  # [batch, frames, joints]

#         # Reshape for LSTM: [batch, seq_len, features]
#         x_reshaped = x.view(batch_size, self.num_frames, -1)

#         # Process sequence with LSTM
#         lstm_out, (h_n, _) = self.encoder_lstm(x_reshaped)

#         # Get final hidden state
#         h_n = h_n[-2:].transpose(0, 1).contiguous().view(batch_size, -1)  # Combine bidirectional

#         # Apply temporal attention
#         context, _ = self.temporal_attention(h_n, lstm_out)

#         # Get joint features from middle frame
#         sample_frame_idx = self.num_frames // 2
#         joint_input = x[:, sample_frame_idx]
#         joint_mask_sample = joint_mask[:, sample_frame_idx]  # [batch, joints]

#         # Project to joint feature space for attention
#         joint_features = self.joint_feature_extractor(joint_input)

#         # Apply joint attention with mask
#         joint_context, _ = self.joint_attention(joint_features, joint_mask_sample)

#         # Flatten and project joint context
#         joint_context = joint_context.reshape(batch_size, -1)
#         joint_context = nn.Linear(self.num_joints * 64, self.hidden_size).to(x.device)(joint_context)

#         # Concatenate temporal context, joint context and condition information
#         combined = torch.cat([context, joint_context, condition_enc], dim=1)

#         # Encode to latent space
#         h = self.fc_encoder[0](combined)  # First layer
#         h = self.fc_encoder[1](h)         # ReLU
#         h = self.fc_encoder[2](h)         # Second layer
#         h = self.fc_encoder[3](h)         # ReLU

#         # Get latent parameters
#         mu = self.mu(h)
#         logvar = self.logvar(h)

#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             return eps * std + mu
#         else:
#             # During inference, just return mean for deterministic output
#             return mu

#     def decode(self, z, condition, original_x=None):
#         batch_size = z.shape[0]
        
#         # Process condition for decoder
#         condition_dec = self.condition_decoder(condition)  # [batch, 64]
        
#         # Create joint mask if original_x is provided (for inference)
#         joint_mask = None
#         if original_x is not None:
#             joint_mask = self.create_joint_mask(original_x)

#         # Combine latent with condition
#         z_conditioned = torch.cat([z, condition_dec], dim=1)
        
#         # Process through FC layers
#         h = self.fc_decoder(z_conditioned)  # [batch, hidden_size]

#         # Initialize decoder hidden state
#         h_0 = torch.zeros(2, batch_size, self.hidden_size).to(z.device)
#         c_0 = torch.zeros(2, batch_size, self.hidden_size).to(z.device)

#         # Prepare decoder input with condition
#         # Repeat condition for each frame
#         condition_seq = condition_dec.unsqueeze(1).repeat(1, self.num_frames, 1)  # [batch, frames, 64]
        
#         # Repeat hidden state for each frame
#         hidden_seq = h.unsqueeze(1).repeat(1, self.num_frames, 1)  # [batch, frames, hidden]
        
#         # Concatenate for LSTM input
#         decoder_input = torch.cat([hidden_seq, condition_seq], dim=2)  # [batch, frames, hidden+64]

#         # Process with decoder LSTM
#         lstm_out, _ = self.decoder_lstm(decoder_input, (h_0, c_0))

#         # Create output tensor
#         output = torch.zeros(batch_size, self.num_frames, self.num_joints, self.coord_dim).to(z.device)

#         # Generate each frame
#         for t in range(self.num_frames):
#             # Get the hidden state for this frame
#             frame_hidden = lstm_out[:, t]
            
#             # Get frame-specific mask if available
#             frame_mask = None
#             if joint_mask is not None:
#                 frame_mask = joint_mask[:, t]

#             # Project to joint features
#             joint_features = self.joint_projection(frame_hidden).view(batch_size, self.num_joints, 64)

#             # Apply joint attention with mask
#             attended_joints, _ = self.joint_attention(joint_features, frame_mask)

#             # Project to coordinates
#             coords = self.output_fc(attended_joints)  # [batch_size, num_joints, 2]

#             # If we have mask, zero out invalid joints
#             if frame_mask is not None:
#                 coords = coords * frame_mask.unsqueeze(-1)

#             # Store in the output tensor
#             output[:, t] = coords

#         return output

#     def forward(self, x, condition):
#         x = x.to(torch.float32)
#         condition = condition.to(torch.float32)
        
#         # Encode
#         mu, logvar = self.encode(x, condition)

#         # Sample latent vector
#         z = self.reparameterize(mu, logvar)

#         # Decode (pass condition explicitly)
#         recon = self.decode(z, condition, x)

#         return recon, mu, logvar, x


# # Modified loss function with KL annealing and free bits
# def loss_function(x, recon, mu, logvar, beta=1.0, free_bits=1e-4):
#     # Create mask for non-zero joints
#     mask = (x.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
    
#     # Apply mask to reconstruction loss (only consider non-zero joints)
#     recon_diff = ((recon - x) ** 2) * mask
#     recon_loss = recon_diff.sum() / mask.sum().clamp(min=1.0)  # Normalize by number of valid points
    
#     # KL divergence with free bits to prevent posterior collapse
#     kld_elements = 1 + logvar - mu.pow(2) - logvar.exp()
#     kld_elements = torch.max(-0.5 * kld_elements, 
#                               torch.ones_like(kld_elements) * -0.5 * free_bits)
#     kld = kld_elements.sum()
    
#     # Total loss
#     return recon_loss, beta * kld


# # Modified KL annealing schedule
# def beta_schedule(epoch, total_epochs, beta_min=0.01, beta_max=1.0, warmup_fraction=0.2):
#     warmup_epochs = int(total_epochs * warmup_fraction)
#     if epoch < warmup_epochs:
#         # Slower warmup to prevent posterior collapse
#         return beta_min + (beta_max - beta_min) * ((epoch / warmup_epochs) ** 2)
#     else:
#         # Cyclical annealing after warmup
#         cycle_progress = (epoch - warmup_epochs) % (total_epochs - warmup_epochs)
#         cycle_fraction = cycle_progress / (total_epochs - warmup_epochs)
#         return beta_min + 0.5 * (beta_max - beta_min) * (1 - np.cos(np.pi * cycle_fraction))



import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class TemporalEncoder(nn.Module):
    def __init__(self, input_shape, nhid=64, cond_dim=50):
        super(TemporalEncoder, self).__init__()
        # input_shape: (30, 29, 2)
        self.temporal_conv = nn.Sequential(
            # First temporal block
            nn.Conv1d(29*2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Second temporal block with downsampling
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Third temporal block
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Fourth temporal block with downsampling
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        # MLP to process temporal features + condition
        self.mlp = nn.Sequential(
            nn.Linear(512 + cond_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.calc_mean = nn.Linear(128, nhid)
        self.calc_logvar = nn.Linear(128, nhid)
        
    def forward(self, x, cond):
        # x shape: [bs, 30, 29, 2]
        bs = x.shape[0]
        x = x.view(bs, 30, -1).transpose(1, 2)  # [bs, 58, 30]
        
        temporal_features = self.temporal_conv(x)  # [bs, 512, 8]
        temporal_features = self.temporal_pool(temporal_features)  # [bs, 512, 1]
        temporal_features = self.flatten(temporal_features)  # [bs, 512]
        
        # Concatenate condition
        features = torch.cat([temporal_features, cond], dim=1)
        features = self.mlp(features)
        
        mean = self.calc_mean(features)
        logvar = self.calc_logvar(features)
        
        return mean, logvar

class TemporalDecoder(nn.Module):
    def __init__(self, output_shape, nhid=64, cond_dim=50):
        super(TemporalDecoder, self).__init__()
        self.output_shape = output_shape  # (30, 29, 2)
        self.nhid = nhid
        self.cond_dim = cond_dim
        
        # Initial processing of latent + condition
        self.mlp = nn.Sequential(
            nn.Linear(nhid + cond_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512 * 4),  # Prepare for conv transpose
            nn.BatchNorm1d(512 * 4),
            nn.ReLU(),
        )
        
        # Temporal upsampling with precise dimension calculation
        self.temporal_upconv = nn.Sequential(
            # First upconv block (4 -> 8)
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Second upconv block (8 -> 16)
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Third upconv block (16 -> 32)
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Adjust to exact 30 frames (32 -> 30)
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),  # Removes 2 frames
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Final output layer
            nn.Conv1d(64, 29*2, kernel_size=1, stride=1),  # [bs, 58, 30]
        )
        
    def forward(self, z, cond):
        bs = z.shape[0]
        if cond is not None:
            z = torch.cat([z, cond], dim=1)
            
        # Process through MLP
        features = self.mlp(z)  # [bs, 512*4]
        features = features.view(bs, 512, 4)  # [bs, 512, 4]
        
        # Temporal upsampling
        output = self.temporal_upconv(features)  # [bs, 58, 30]
        
        # Reshape to original format
        output = output.transpose(1, 2)  # [bs, 30, 58]
        output = output.reshape(bs, *self.output_shape)  # [bs, 30, 29, 2]
        
        return output

class SignLanguageVAE(nn.Module):
    def __init__(self, input_shape, nhid=64, cond_dim=50):
        super(SignLanguageVAE, self).__init__()
        self.input_shape = input_shape
        self.nhid = nhid
        self.cond_dim = cond_dim
        
        self.encoder = TemporalEncoder(input_shape, nhid, cond_dim)
        self.decoder = TemporalDecoder(input_shape, nhid, cond_dim)
        
    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(mean.device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma
    
    def forward(self, x, cond):
        mean, logvar = self.encoder(x, cond)
        z = self.sampling(mean, logvar)
        x_hat = self.decoder(z, cond)
        return x_hat, mean, logvar
    
    def generate(self, cond):
        batch_size = cond.shape[0]
        z = torch.randn((batch_size, self.nhid)).to(cond.device)
        return self.decoder(z, cond)


def loss_function(x, x_hat, mean, logvar, beta=0.1):
    # Reconstruction loss (MSE for coordinates)
    reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    # KL divergence
    KL_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    # Total loss with beta-VAE to encourage better disentanglement
    total_loss = reconstruction_loss + beta * KL_divergence
    
    return total_loss, reconstruction_loss, KL_divergence