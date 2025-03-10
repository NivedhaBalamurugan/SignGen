import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, Input, Reshape, TimeDistributed, Bidirectional, UpSampling1D, Concatenate, SpectralNormalization, Add
from keras.models import Model
from keras.layers import GRU, Dense, Input, Dropout, BatchNormalization, Reshape, Conv1D, Conv2D, LeakyReLU, Permute, GaussianNoise
from config import *

def self_attention(x):
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    return tf.keras.layers.Add()([x, attn])

def build_generator():
    # Input: noise + word embedding
    inputs = Input(shape=(CGAN_NOISE_DIM + EMBEDDING_DIM,))
    
    # Dense layers to process combined input
    x = Dense(256, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Reshape to prepare for recurrent layers
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Reshape((16, 32))(x)  # Reshape to sequence format
    
    # Remove self-attention from GRU layers
    x1 = Bidirectional(GRU(64, return_sequences=True))(x)
    x2 = Bidirectional(GRU(64, return_sequences=True))(x1)
    x2 = Add()([x1, x2])

    # Apply self-attention **after upsampling**, before final dense layers
    x = UpSampling1D(2)(x2)  # Expand frames
    x = self_attention(x)   # Apply here to enhance movement features

    # Expand to full sequence length
    x = TimeDistributed(Dense(128, activation="tanh"))(x)
    x = UpSampling1D(2)(x)
    x = self_attention(x)

    x = Add()([x2, x])

    # Final dense layers to generate coordinates
    x = TimeDistributed(Dense(NUM_JOINTS * NUM_COORDINATES, activation="tanh"))(x)
    outputs = Reshape((MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES))(x[:, :MAX_FRAMES])
    
    return Model(inputs, outputs)

def build_discriminator():
    # Separate inputs for skeleton and word embeddings
    skeleton_input = Input(shape=(MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES))
    word_input = Input(shape=(EMBEDDING_DIM,))
    
    # Process skeleton with 1D convolutions
    x = Reshape((MAX_FRAMES, NUM_JOINTS * NUM_COORDINATES))(skeleton_input)

    x = GaussianNoise(0.05)(x) 
    
    # Process joint relationships with 1D convolutions
    x = Conv1D(128, kernel_size=5, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv1D(256, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    # Further process skeleton data
    x = Conv1D(256, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    # GRU layers for temporal processing
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = Bidirectional(GRU(128, return_sequences=False))(x)
    
    # Process word embedding
    word_features = SpectralNormalization(Dense(128, activation="relu"))(word_input)
    word_features = BatchNormalization()(word_features)
    
    # Concatenate skeleton features with word features
    combined = Concatenate()([x, word_features])
    
    # Dense classification layers
    x = Dense(256, activation="relu")(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1)(x)
    
    return Model([skeleton_input, word_input], outputs)

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def discriminator_loss(real_output, fake_output):
    # Wasserstein loss for discriminator
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    return real_loss + fake_loss

def bone_length_consistency_loss(skeletons, joint_connections):
    """
    Penalize variations in bone length across frames
    joint_connections: list of tuples (joint_idx1, joint_idx2) defining bones
    """
    losses = []
    for i in range(len(joint_connections)):
        joint1_idx, joint2_idx = joint_connections[i]
        
        # Get positions of connected joints
        joint1 = skeletons[:, :, joint1_idx, :]  # [batch, frames, coordinates]
        joint2 = skeletons[:, :, joint2_idx, :]  # [batch, frames, coordinates]
        
        # Calculate bone lengths for each frame
        bone_lengths = tf.sqrt(tf.reduce_sum(tf.square(joint1 - joint2), axis=-1) + 1e-8)  # [batch, frames]
        
        # Calculate variance of bone length across frames (should be near 0)
        mean_lengths = tf.reduce_mean(bone_lengths, axis=1, keepdims=True)  # [batch, 1]
        length_variance = tf.reduce_mean(tf.square(bone_lengths - mean_lengths))
        
        losses.append(length_variance)
    
    return tf.reduce_mean(losses)

def motion_smoothness_loss(skeletons):
    """
    Encourage smooth motion while ensuring hands move enough, while preventing NaNs.
    """
    velocity = skeletons[:, 1:] - skeletons[:, :-1]  # First derivative (motion between frames)
    acceleration = velocity[:, 1:] - velocity[:, :-1]  # Second derivative (jerkiness)

    # Prevent NaN by adding a small epsilon before squaring
    velocity_loss = tf.reduce_mean(tf.square(tf.clip_by_value(velocity, -1.0, 1.0) + 1e-6))
    acceleration_loss = tf.reduce_mean(tf.square(tf.clip_by_value(acceleration, -1.0, 1.0) + 1e-6))

    # Ensure variance doesn't lead to NaN
    variance_loss = motion_variance_loss(skeletons) + 1e-6

    return (50 * velocity_loss + 50 * acceleration_loss) - (100 * variance_loss)

def anatomical_plausibility_loss(skeletons, joint_angle_limits):
    """
    Penalizes anatomically implausible joint angles
    """
    losses = []
    
    for (joint1, joint2, joint3, min_angle, max_angle) in joint_angle_limits:
        # Get joint positions
        pos1 = skeletons[:, :, joint1, :]  # [batch, frames, coordinates]
        pos2 = skeletons[:, :, joint2, :]  # [batch, frames, coordinates]
        pos3 = skeletons[:, :, joint3, :]  # [batch, frames, coordinates]
        
        # Calculate vectors
        v1 = pos1 - pos2  # Vector from joint2 to joint1
        v2 = pos3 - pos2  # Vector from joint2 to joint3
        
        # Calculate dot product
        dot_product = tf.reduce_sum(v1 * v2, axis=-1)
        
        # Calculate magnitudes
        v1_mag = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=-1) + 1e-8)
        v2_mag = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=-1) + 1e-8)
        
        # Calculate cosine of angle
        cos_angle = dot_product / (v1_mag * v2_mag)
        cos_angle = tf.clip_by_value(cos_angle, -0.9999, 0.9999)  # Prevent numerical issues
        
        # Convert to angle in radians
        angle = tf.acos(cos_angle)
        
        # Penalize angles outside the defined limits
        min_rad = min_angle * np.pi / 180
        max_rad = max_angle * np.pi / 180
        
        penalty = tf.maximum(0.0, min_rad - angle) + tf.maximum(0.0, angle - max_rad)
        losses.append(tf.reduce_mean(penalty))
    
    return tf.reduce_mean(losses)

def semantic_consistency_loss(generated_skeletons, word_vectors):
    """
    Ensures that similar words produce similar motions
    """
    # Calculate pairwise cosine similarity between word vectors
    word_similarities = tf.matmul(
        tf.nn.l2_normalize(word_vectors, axis=1),
        tf.nn.l2_normalize(word_vectors, axis=1),
        transpose_b=True
    )
    
    # Calculate motion similarity (use average joint positions)
    flattened_skeletons = tf.reshape(generated_skeletons, 
                                    (tf.shape(generated_skeletons)[0], -1))
    skeleton_similarities = tf.matmul(
        tf.nn.l2_normalize(flattened_skeletons, axis=1),
        tf.nn.l2_normalize(flattened_skeletons, axis=1),
        transpose_b=True
    )
    
    # Loss is higher when word similarity doesn't match motion similarity
    return tf.reduce_mean(tf.square(word_similarities - skeleton_similarities))

def motion_variance_loss(skeletons):
    """
    Encourage movement by maximizing per-joint variance across frames.
    """

    hand_joints = [x for x in range(7, 29)] 
    hand_frames = tf.gather(skeletons, hand_joints, axis=2)

    # Compute variance of hand motion across frames
    variance = tf.math.reduce_std(hand_frames, axis=1)

    # Maximize variance to encourage movement
    return -tf.reduce_mean(variance) * 10
