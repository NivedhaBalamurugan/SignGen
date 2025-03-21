import torch
import torch.nn as nn
from config import *

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
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_prob=0.5):  # Increased dropout
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2)
        self.attention = Attention(hidden_dim * 2)
        self.fc_mu = nn.Linear(hidden_dim * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 4, latent_dim)

        self.batch_norm_hidden = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout = nn.Dropout(p=dropout_prob)  # Increased dropout

    def forward(self, x):
        B, T, _, _ = x.size()
        x = x.view(B, T, -1)

        mask = (x.sum(dim=2) != 0).float()
        outputs, (h_n, _) = self.lstm(x)

        outputs = self.dropout(outputs)  # Dropout after LSTM

        outputs = outputs.permute(0, 2, 1)
        outputs = self.batch_norm_hidden(outputs)
        outputs = outputs.permute(0, 2, 1)

        context, attn_weights = self.attention(outputs, mask)

        mu = self.fc_mu(torch.cat([outputs.mean(dim=1), context], dim=-1))
        logvar = self.fc_logvar(torch.cat([outputs.mean(dim=1), context], dim=-1))

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        return z, mu, logvar, attn_weights


class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim, output_dim, seq_len, dropout_prob=0.5):  # Increased dropout
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim + cond_dim, latent_dim + cond_dim)
        self.lstm = nn.LSTM(latent_dim + cond_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)  # Increased dropout

    def forward(self, z, cond):
        B = z.size(0)
        z_cond = torch.cat([z, cond], dim=-1)
        z_cond = torch.relu(self.fc(z_cond))

        z_seq = z_cond.unsqueeze(1).repeat(1, self.seq_len, 1)
        outputs, _ = self.lstm(z_seq)

        outputs = self.dropout(outputs)  # Dropout after LSTM

        outputs = outputs.permute(0, 2, 1)
        outputs = self.batch_norm(outputs)
        outputs = outputs.permute(0, 2, 1)

        outputs = self.dropout(outputs)  # Dropout before output layer

        recon_seq = torch.tanh(self.fc_out(outputs))
        return recon_seq

class LatentClassifier(nn.Module):
    def __init__(self, latent_dim, dropout_prob=0.5):  # Added dropout
        super(LatentClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),  # Added dropout
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.fc(z)

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dim, output_dim, seq_len):
        super(ConditionalVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout_prob=0.5)  # Increased dropout
        self.decoder = Decoder(latent_dim, cond_dim, hidden_dim, output_dim, seq_len, dropout_prob=0.5)  # Increased dropout
        self.latent_classifier = LatentClassifier(latent_dim, dropout_prob=0.5)  # Added dropout

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