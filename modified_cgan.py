import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, Input, Reshape, TimeDistributed, Bidirectional, UpSampling1D
from keras.models import Model
from keras.layers import GRU, Dense, Input, Dropout, BatchNormalization, Reshape, Conv1D, Conv2D, LeakyReLU, Permute
from keras.layers import Concatenate, Add, Lambda
from config import *

def build_generator():
    # Separate inputs for noise and word embedding
    noise_input = Input(shape=(CGAN_NOISE_DIM,))
    word_input = Input(shape=(EMBEDDING_DIM,))
    
    # Process noise
    n = Dense(128, activation="relu")(noise_input)
    n = BatchNormalization()(n)
    
    # Process word embedding separately
    w = Dense(128, activation="relu")(word_input)
    w = BatchNormalization()(w)
    
    # Combine them with concatenation
    x = Concatenate()([n, w])
    
    # First dense layer with skip connection from word embedding
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Add a skip connection from word embedding
    w2 = Dense(256, activation="linear")(word_input)
    x = Add()([x, w2])
    
    # Second dense layer with skip connection
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    
    # Add another skip connection
    w3 = Dense(512, activation="linear")(word_input)
    x = Add()([x, w3])
    
    # Reshape to prepare for recurrent layers
    x = Reshape((16, 32))(x)  # Reshape to sequence format
    
    # Word embedding as additional input to GRU layers
    # Reshape word embedding for concatenation with sequence
    w_seq = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, 16, 1]))(word_input)
    
    # First Bidirectional GRU with word info
    x = Concatenate(axis=2)([x, w_seq])
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    
    # Add word info again before second GRU
    x = Concatenate(axis=2)([x, w_seq])
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    
    # Expand to full sequence length
    x = TimeDistributed(Dense(128, activation="relu"))(x)
    x = UpSampling1D(2)(x)  # From 16 to 32 frames
    
    # Add temporal noise to prevent identical frames
    frame_noise = Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0, stddev=0.01))(x)
    x = frame_noise
    
    # Final dense layers with word conditioning
    # Reshape word embedding for the expanded sequence length
    w_seq_full = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, 32, 1]))(word_input)
    x = Concatenate(axis=2)([x, w_seq_full[:, :MAX_FRAMES]])
    
    # Final output layer
    x = TimeDistributed(Dense(NUM_JOINTS * NUM_COORDINATES, activation="tanh"))(x)
    outputs = Reshape((MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES))(x[:, :MAX_FRAMES])
    
    return Model([noise_input, word_input], outputs)

def build_discriminator():
    # Input: skeleton + word embeddings
    skeleton_input = Input(shape=(MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES))
    word_input = Input(shape=(EMBEDDING_DIM,))
    
    # Process skeleton
    # Reshape for easier processing
    x = Reshape((MAX_FRAMES, NUM_JOINTS * NUM_COORDINATES))(skeleton_input)
    
    # Create word embedding expanded across time
    word_expanded = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, MAX_FRAMES, 1]))(word_input)
    
    # Concatenate word with skeleton at the sequence level
    x = Concatenate(axis=2)([x, word_expanded])
    
    # 1D convolutions across time dimension
    x = Conv1D(128, kernel_size=5, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv1D(256, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    # Process joint relationships with 1D convolutions
    x = Conv1D(256, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    # Add word embedding again before GRU
    # Reshape word for the current sequence length
    current_seq_len = tf.shape(x)[1]
    word_expanded_2 = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, current_seq_len, 1]))(word_input)
    x = Concatenate(axis=2)([x, word_expanded_2])
    
    # GRU layers for temporal processing
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    
    # Add word embedding again
    word_expanded_3 = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, current_seq_len, 1]))(word_input)
    x = Concatenate(axis=2)([x, word_expanded_3])
    x = Bidirectional(GRU(128, return_sequences=False))(x)
    
    # Concatenate the final word embedding
    x = Concatenate()([x, word_input])
    
    # Dense classification layers
    x = Dense(256, activation="relu")(x)
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
    Penalize jerky, non-smooth motion between frames
    """
    # Calculate velocity (first derivative of position)
    velocity = skeletons[:, 1:] - skeletons[:, :-1]  # [batch, frames-1, joints, coords]
    
    # Calculate acceleration (second derivative of position)
    acceleration = velocity[:, 1:] - velocity[:, :-1]  # [batch, frames-2, joints, coords]
    
    # Penalize large accelerations (encourage smooth motion)
    return tf.reduce_mean(tf.square(acceleration))

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

def temporal_diversity_loss(generated_skeletons):
    """
    Encourages diversity between frames
    """
    # Calculate frame-to-frame differences
    frame_diffs = generated_skeletons[:, 1:] - generated_skeletons[:, :-1]
    
    # Calculate magnitude of differences
    diff_magnitudes = tf.sqrt(tf.reduce_sum(tf.square(frame_diffs), axis=[2, 3]) + 1e-8)
    
    # Penalize when frames are too similar (encourage movement)
    min_movement = 0.01  # Minimum expected movement between frames
    movement_penalty = tf.maximum(0.0, min_movement - diff_magnitudes)
    
    return tf.reduce_mean(movement_penalty)

def diversity_loss(generated_skeletons):
    """
    Encourages diversity across the batch
    """
    batch_size = tf.shape(generated_skeletons)[0]
    if batch_size <= 1:
        return 0.0
        
    # Flatten each sample
    flattened = tf.reshape(generated_skeletons, [batch_size, -1])
    
    # Normalize
    normalized = tf.nn.l2_normalize(flattened, axis=1)
    
    # Calculate similarity matrix
    similarity = tf.matmul(normalized, normalized, transpose_b=True)
    
    # Remove diagonal (self-similarity)
    mask = tf.ones_like(similarity) - tf.eye(batch_size)
    masked_similarity = similarity * mask
    
    # Calculate mean similarity (lower is more diverse)
    mean_similarity = tf.reduce_sum(masked_similarity) / (tf.cast(batch_size, tf.float32) * (tf.cast(batch_size, tf.float32) - 1))
    
    return mean_similarity  # We want to minimize this
