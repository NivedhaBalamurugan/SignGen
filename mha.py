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


def kalman_filter_sequence(fused_seq):
    num_frames, num_joints, _ = fused_seq.shape
    smoothed_seq = np.zeros_like(fused_seq)
    
    dt = 1.0  # Assume a time step of 1 between consecutive frames.
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])
    
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    
    q = 0.05  # A bit higher than 0.01
    r = 0.05  # Lower than 0.1 for less smoothing lag
    Q = q * np.eye(4)
    R = r * np.eye(2)
        
    for j in range(num_joints):
        x = np.array([fused_seq[0, j, 0],
                      fused_seq[0, j, 1],
                      0,
                      0]).reshape(4, 1)
        P = np.eye(4)
        
        smoothed_seq[0, j, :] = fused_seq[0, j, :]
        
        for t in range(1, num_frames):
            x_pred = A @ x
            P_pred = A @ P @ A.T + Q
            
            z = fused_seq[t, j, :].reshape(2, 1)
            
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            x = x_pred + K @ (z - H @ x_pred)
            P = (np.eye(4) - K @ H) @ P_pred
            
            smoothed_seq[t, j, :] = x[:2, 0]
            
    return smoothed_seq

import numpy as np
from scipy import interpolate

def joint_trajectory_smoothing_with_bezier(fused_sequence, velocity_threshold=0.07, smoothing_factor=0.6):
    """
    Apply Joint Trajectory Smoothing with Bezier Curve Optimization on a skeleton sequence.
    
    Parameters:
    -----------
    fused_sequence : numpy.ndarray
        The input fused skeleton sequence with shape (num_frames, num_joints, 2)
    velocity_threshold : float
        Threshold for detecting discontinuities in joint velocity
    smoothing_factor : float
        Controls how much smoothing to apply (0.0 to 1.0)
        
    Returns:
    --------
    numpy.ndarray
        Smoothed skeleton sequence with the same shape as input
    """
    num_frames, num_joints, coords = fused_sequence.shape
    smoothed_sequence = np.copy(fused_sequence)
    
    # Process each joint independently
    for joint_id in range(num_joints):
        # Extract trajectory for this joint across all frames
        joint_trajectory = fused_sequence[:, joint_id, :]
        
        # Calculate velocities between frames
        velocities = np.sqrt(np.sum(np.diff(joint_trajectory, axis=0)**2, axis=1))
        
        # Detect discontinuities (frames where velocity changes suddenly)
        velocity_changes = np.abs(np.diff(velocities))
        discontinuity_indices = np.where(velocity_changes > velocity_threshold)[0] + 1
        
        # If discontinuities found, apply Bezier smoothing
        if len(discontinuity_indices) > 0:
            for idx in discontinuity_indices:
                if idx > 1 and idx < num_frames - 2:
                    # Define window around discontinuity (2 frames before and after)
                    window_start = max(0, idx - 2)
                    window_end = min(num_frames, idx + 3)
                    window_size = window_end - window_start
                    
                    # Skip if window too small
                    if window_size < 4:
                        continue
                    
                    # Get original trajectory in window
                    original_points = joint_trajectory[window_start:window_end]
                    
                    # Create parameter t for curve fitting
                    t = np.linspace(0, 1, window_size)
                    
                    # Fit Bezier curve (for x and y coordinates separately)
                    for coord in range(coords):
                        # Create control points for Bezier curve
                        x = np.linspace(0, 1, window_size)
                        y = original_points[:, coord]
                        
                        # Calculate control points using De Casteljau's algorithm
                        # For a cubic Bezier curve
                        if window_size >= 4:
                            # Fit a cubic spline through the points
                            tck = interpolate.splrep(x, y, s=smoothing_factor)
                            
                            # Generate smoothed points
                            y_smooth = interpolate.splev(x, tck)
                            
                            # Apply smoothing with original points for stability
                            blend = np.linspace(0.2, 0.8, window_size)
                            y_final = y * (1 - blend) + y_smooth * blend
                            
                            # Update the trajectory
                            smoothed_sequence[window_start:window_end, joint_id, coord] = y_final
    
    # Ensure velocity consistency across the entire sequence
    for joint_id in range(num_joints):
        joint_trajectory = smoothed_sequence[:, joint_id, :]
        
        # Calculate current velocities
        velocities = np.diff(joint_trajectory, axis=0)
        
        # Find any remaining high velocities
        velocity_magnitudes = np.sqrt(np.sum(velocities**2, axis=1))
        high_velocity_indices = np.where(velocity_magnitudes > velocity_threshold * 2)[0]
        
        # Dampen extreme velocities
        for idx in high_velocity_indices:
            # Reduce the velocity
            dampened_velocity = velocities[idx] * 0.7
            # Apply the dampened velocity
            smoothed_sequence[idx+1, joint_id] = smoothed_sequence[idx, joint_id] + dampened_velocity
    
    # Enforce joint dependencies (ensure connected joints maintain physical constraints)
    # Define joint connections based on skeleton structure
    # Example connections for upper body skeleton (customize based on your joint mapping)
    joint_connections = [
        (0, 1), (1, 2), (2, 3),  # Right arm chain
        (0, 4), (4, 5), (5, 6),  # Left arm chain
        # Add more connections based on your skeleton structure
    ]
    
    for frame in range(num_frames):
        for joint1, joint2 in joint_connections:
            if joint1 < num_joints and joint2 < num_joints:
                # Get current distance between connected joints
                current_dist = np.linalg.norm(smoothed_sequence[frame, joint1] - smoothed_sequence[frame, joint2])
                
                # Get original distance between these joints
                original_dist = np.linalg.norm(fused_sequence[frame, joint1] - fused_sequence[frame, joint2])
                
                # If distance has changed significantly, adjust to maintain constraints
                if abs(current_dist - original_dist) > 0.05 * original_dist:
                    # Calculate the direction vector
                    direction = smoothed_sequence[frame, joint2] - smoothed_sequence[frame, joint1]
                    direction = direction / np.linalg.norm(direction)
                    
                    # Adjust joint2 position to maintain proper distance
                    smoothed_sequence[frame, joint2] = smoothed_sequence[frame, joint1] + direction * original_dist
    
    return smoothed_sequence


def fuse_sequences(input_word, isSave=True):
    vae_sequence = new_cvae_inf.get_cvae_sequence(input_word, isSave)  
    gan_sequence = cgan_inference.get_cgan_sequence(input_word, isSave)

    gan_body = gan_sequence[:, :7, :]    
    vae_hands = vae_sequence[:, 7:, :]   
    
    fused_sequence_bef_enh = np.concatenate([gan_body, vae_hands], axis=1)  
    
    fused_sequence = joint_trajectory_smoothing_with_bezier(fused_sequence_bef_enh)
    # fused_sequence = kalman_filter_sequence(fused_sequence_bef_enh)

    if isSave:
        show_output.save_generated_sequence(fused_sequence_bef_enh, MHA_OUTPUT_FRAMES_BEF_ENH, MHA_OUTPUT_VIDEO)
        show_output.save_generated_sequence(fused_sequence, MHA_OUTPUT_FRAMES, MHA_OUTPUT_VIDEO)
    return fused_sequence


if __name__ == "__main__":
    fuse_sequences("police", isSave=True)