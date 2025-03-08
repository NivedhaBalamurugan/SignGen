import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, Input, RepeatVector, Reshape, TimeDistributed, Bidirectional, UpSampling1D
from keras.models import Model
from keras.layers import GRU, Dense, Input, Flatten, Dropout, BatchNormalization, Reshape, RepeatVector
from config import *

def build_generator():
    # Input: noise + word embedding
    inputs = Input(shape=(CGAN_NOISE_DIM + 20,))
    
    # Dense layers to process combined input
    x = Dense(256, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Reshape to prepare for recurrent layers
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Reshape((16, 32))(x)  # Reshape to sequence format
    
    # Bidirectional GRU for temporal coherence
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    
    # Expand to full sequence length
    x = TimeDistributed(Dense(128, activation="relu"))(x)
    x = UpSampling1D(2)(x)  # From 16 to 32 frames (can trim later)
    
    # Final dense layers to generate coordinates
    x = TimeDistributed(Dense(NUM_JOINTS * NUM_COORDINATES, activation="tanh"))(x)
    outputs = Reshape((MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES))(x[:, :MAX_FRAMES])  # Ensure output size matches expected
    
    return Model(inputs, outputs)

def build_discriminator():
    # Input: skeleton + word embeddings
    input_shape = (MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES + EMBEDDING_DIM)
    inputs = Input(shape=input_shape)
    
    # Reshape for easier processing
    x = Reshape((MAX_FRAMES, NUM_JOINTS * (NUM_COORDINATES + EMBEDDING_DIM)))(inputs)
    
    # Temporal feature extraction
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Bidirectional(GRU(64, return_sequences=False))(x)  # Return only final state
    
    # Dense classification layers
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)  # No activation for Wasserstein Loss
    
    return Model(inputs, outputs)

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