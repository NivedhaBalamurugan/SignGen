import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *  
import new_cvae_inf
import numpy as np
import show_output
import cgan_inference
import os
from scipy import interpolate

class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):  
        super(MultiHeadAttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        assert feature_dim % num_heads == 0, f"Feature dimension ({feature_dim}) must be divisible by number of heads ({num_heads})"
        self.head_dim = feature_dim // num_heads
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(feature_dim, feature_dim)
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, gan_seq, vae_seq):
        batch_size, seq_len, _ = gan_seq.shape
        
        q = self.query(gan_seq).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(vae_seq).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(vae_seq).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.feature_dim)
        
        output = self.output_layer(context)
        
        fused_seq = self.alpha * gan_seq + (1 - self.alpha) * output
        
        return fused_seq

class SignSequenceFuser(nn.Module):
    def __init__(self, feature_dim, num_layers=2, num_heads=2):  
        super(SignSequenceFuser, self).__init__()
        self.layers = nn.ModuleList([
            MultiHeadAttentionFusion(feature_dim, num_heads=num_heads) for _ in range(num_layers)
        ])
        
        self.feature_refinement = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, gan_seq, vae_seq):
        fused_seq = self.layers[0](gan_seq, vae_seq)
        
        for layer in self.layers[1:]:
            fused_seq = layer(fused_seq, vae_seq)
            
        output = self.feature_refinement(fused_seq)
        
        final_output = output + 0.5 * (gan_seq + vae_seq)
        
        return final_output

def joint_trajectory_smoothing_with_bezier(fused_sequence, velocity_threshold=0.07, smoothing_factor=0.6):
   
    num_frames, num_joints, coords = fused_sequence.shape
    smoothed_sequence = np.copy(fused_sequence)
    
    for joint_id in range(num_joints):
        joint_trajectory = fused_sequence[:, joint_id, :]
        
        velocities = np.sqrt(np.sum(np.diff(joint_trajectory, axis=0)**2, axis=1))
        
        velocity_changes = np.abs(np.diff(velocities))
        discontinuity_indices = np.where(velocity_changes > velocity_threshold)[0] + 1
        
        if len(discontinuity_indices) > 0:
            for idx in discontinuity_indices:
                if idx > 1 and idx < num_frames - 2:
                    window_start = max(0, idx - 2)
                    window_end = min(num_frames, idx + 3)
                    window_size = window_end - window_start
                    
                    if window_size < 4:
                        continue
                    
                    original_points = joint_trajectory[window_start:window_end]
                    
                    t = np.linspace(0, 1, window_size)
                    
                    for coord in range(coords):
                        x = np.linspace(0, 1, window_size)
                        y = original_points[:, coord]
                        
                        if window_size >= 4:
                            tck = interpolate.splrep(x, y, s=smoothing_factor)
                            
                            y_smooth = interpolate.splev(x, tck)
                            
                            blend = np.linspace(0.2, 0.8, window_size)
                            y_final = y * (1 - blend) + y_smooth * blend
                            
                            smoothed_sequence[window_start:window_end, joint_id, coord] = y_final
    
    for joint_id in range(num_joints):
        joint_trajectory = smoothed_sequence[:, joint_id, :]
        
        velocities = np.diff(joint_trajectory, axis=0)
        
        velocity_magnitudes = np.sqrt(np.sum(velocities**2, axis=1))
        high_velocity_indices = np.where(velocity_magnitudes > velocity_threshold * 2)[0]
        
        for idx in high_velocity_indices:
            dampened_velocity = velocities[idx] * 0.7
            smoothed_sequence[idx+1, joint_id] = smoothed_sequence[idx, joint_id] + dampened_velocity
    
    joint_connections = [
        (0, 1), (1, 2), (2, 3),  
        (0, 4), (4, 5), (5, 6), 
    ]
    
    for frame in range(num_frames):
        for joint1, joint2 in joint_connections:
            if joint1 < num_joints and joint2 < num_joints:
                current_dist = np.linalg.norm(smoothed_sequence[frame, joint1] - smoothed_sequence[frame, joint2])
                
                original_dist = np.linalg.norm(fused_sequence[frame, joint1] - fused_sequence[frame, joint2])
                
                if abs(current_dist - original_dist) > 0.05 * original_dist:
                    direction = smoothed_sequence[frame, joint2] - smoothed_sequence[frame, joint1]
                    direction = direction / np.linalg.norm(direction)
                    
                    smoothed_sequence[frame, joint2] = smoothed_sequence[frame, joint1] + direction * original_dist
    
    return smoothed_sequence


def fuse_sequences(input_word, isSave=True):
    vae_sequence, ssim_score = new_cvae_inf.get_cvae_sequence(input_word, isSave)  
    gan_sequence, diversity_score = cgan_inference.get_cgan_sequence(input_word, isSave)

    performance = {}
    performance["CVAE_SSIM"] = ssim_score
    performance["CGAN_Diversity"] = diversity_score

    gan_indices = [0, 1, 4, 5, 6]
    vae_indices = [2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

    fused_sequence_bef_enh = np.zeros((gan_sequence.shape[0], len(gan_indices) + len(vae_indices), gan_sequence.shape[2]))

    fused_sequence_bef_enh[:, gan_indices, :] = gan_sequence[:, gan_indices, :]
    fused_sequence_bef_enh[:, vae_indices, :] = vae_sequence[:, vae_indices, :] 
    
    fused_sequence = joint_trajectory_smoothing_with_bezier(fused_sequence_bef_enh)
    
    print(f"Generated Fused sequence for '{input_word}': {fused_sequence.shape}")

    main_word = check_extended_words(input_word.lower())
    fused_ssim = new_cvae_inf.compute_ssim_score(fused_sequence, main_word)
    fused_diversity = cgan_inference.get_diversity_score(main_word, fused_sequence)

    performance["FUSED_SSIM"] = fused_ssim
    performance["FUSED_Diversity"] = fused_diversity

    print("SSIM score for fused sequence after optimization:", fused_ssim)
    print("Diversity score for fused sequence after optimization:", fused_diversity)

    perf_path = os.path.join("Dataset", "performance.json")
    with open(perf_path, 'w') as f:
        json.dump(performance, f)

    if isSave:
        show_output.save_generated_sequence(fused_sequence, MHA_OUTPUT_FRAMES, MHA_OUTPUT_VIDEO, "fused")
    return fused_sequence

def get_average_perf_metrics():
    main_words = get_main_words()
    avg_ssim = 0
    avg_div = 0
    for word in main_words:
        fuse_sequences(word, isSave=False)
        
        perf_path = os.path.join("Dataset", "performance.json")
        with open(perf_path, 'r') as file:
            perf_met = json.load(file)

        avg_ssim += perf_met["FUSED_SSIM"]
        avg_div += perf_met["FUSED_Diversity"]

    avg_ssim /= len(main_words)
    avg_div /= len(main_words)

    print("Average SSIM score for fused sequence ", avg_ssim)
    print("Average Diversity score for fused sequence ", avg_div)


if __name__ == "__main__":
    fuse_sequences("computer", isSave=True)
    # get_average_perf_metrics()