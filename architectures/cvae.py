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