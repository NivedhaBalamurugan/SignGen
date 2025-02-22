import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Dynamic Positional Encoding
class DynamicPositionalEncoding(nn.Module):
    def __init__(self, num_frames, num_joints, embed_dim):
        super(DynamicPositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(num_frames, num_joints, embed_dim))

    def forward(self, x):
        print(f"Adding positional encoding: input shape {x.shape}, encoding shape {self.positional_encoding.shape}")
        return x + self.positional_encoding

# 2. Cross-Attention Mechanism
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        print(f"Applying cross-attention: query shape {query.shape}, key shape {key.shape}, value shape {value.shape}")
        
        query = query.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        attn_output, _ = self.multihead_attn(query, key, value)

        attn_output = attn_output.permute(1, 0, 2)
        print(f"Cross-attention output shape: {attn_output.shape}")
        return attn_output

# 3. Fusion Model
class FusionModel(nn.Module):
    def __init__(self, embed_dim=3, num_heads=3, num_frames=233, num_joints=49):
        super(FusionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.num_joints = num_joints

        self.positional_encoding = DynamicPositionalEncoding(num_frames, num_joints, embed_dim)

        self.cross_attention = CrossAttention(embed_dim, num_heads)

        self.weight_seq1 = nn.Parameter(torch.randn(num_frames, num_joints, 1))
        self.weight_seq2 = nn.Parameter(torch.randn(num_frames, num_joints, 1))

        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, seq1, seq2):
        print(f"Input sequence shapes: seq1 {seq1.shape}, seq2 {seq2.shape}")

        seq1 = self.positional_encoding(seq1)
        seq2 = self.positional_encoding(seq2)

        fused_seq = self.cross_attention(seq1, seq2, seq2)

        weights = torch.softmax(torch.cat([self.weight_seq1, self.weight_seq2], dim=-1), dim=-1)
        print(f"Weights shape: {weights.shape}, min: {weights.min().item()}, max: {weights.max().item()}")

        fused_seq = weights[..., 0:1] * seq1 + weights[..., 1:2] * seq2
        print(f"Fused sequence shape after weighted fusion: {fused_seq.shape}")

        refined_seq = self.fc(fused_seq)
        print(f"Refined sequence shape: {refined_seq.shape}")

        return refined_seq

# 4. Improved Loss Function
class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()

    def forward(self, fused_seq, seq1, seq2):
        print(f"Computing loss: fused_seq {fused_seq.shape}, seq1 {seq1.shape}, seq2 {seq2.shape}")

        reconstruction_loss = F.mse_loss(fused_seq, seq1) + F.mse_loss(fused_seq, seq2)
        print(f"Reconstruction Loss: {reconstruction_loss.item()}")

        temporal_loss = F.mse_loss(fused_seq[1:], fused_seq[:-1])
        print(f"Temporal Consistency Loss: {temporal_loss.item()}")

        spatial_loss = F.mse_loss(fused_seq[:, 1:], fused_seq[:, :-1])
        print(f"Spatial Consistency Loss: {spatial_loss.item()}")

        total_loss = reconstruction_loss + 0.1 * temporal_loss + 0.1 * spatial_loss
        print(f"Total Loss: {total_loss.item()}\n")
        return total_loss

def fuse_sequences(cvae_seq, cgan_seq):
    print("\n===== Running Fusion Model =====")
    
    model = FusionModel()
    criterion = FusionLoss()

    print("Starting forward pass...")
    fused_seq = model(cvae_seq, cgan_seq)
    print("Forward pass complete.")

    print("Computing loss...")
    loss = criterion(fused_seq, cvae_seq, cgan_seq)
    
    print(f"Final Fused Sequence Shape: {fused_seq.shape}")
    print(f"Final Loss: {loss.item()}")
    
    return fused_seq

# if __name__ == "__main__":
#     cvae_seq = torch.randn(233, 49, 3)
#     cgan_seq = torch.randn(233, 49, 3)

#     # Run fusion model
#     fused_seq = fuse_sequences(cvae_seq, cgan_seq)
