import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, Input, RepeatVector, Reshape
from keras.models import Model
from keras.layers import GRU, Dense, Input, Flatten, Dropout, BatchNormalization, Reshape, RepeatVector
from config import *

def build_generator():
    inputs = Input(shape=(CGAN_NOISE_DIM + 20,))  # Noise + one-hot encoded word embeddings (20 dimensions)
    x = Dense(128, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(MAX_FRAMES * NUM_JOINTS * NUM_COORDINATES, activation="tanh")(x)  # Output skeleton
    outputs = Reshape((MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES))(outputs)  # Reshape to (30, 29, 2)
    return Model(inputs, outputs)

def build_discriminator():
    input_shape = (MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES + 20)  # Skeleton + one-hot encoded word embeddings
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
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

# def bone_length_consistency_loss(skeletons, joint_connections):
#     """
#     Penalize variations in bone length across frames
#     joint_connections: list of tuples (joint_idx1, joint_idx2) defining bones
#     """
#     losses = []
#     for i in range(len(joint_connections)):
#         joint1_idx, joint2_idx = joint_connections[i]
        
#         # Get positions of connected joints
#         joint1 = skeletons[:, :, joint1_idx, :]  # [batch, frames, coordinates]
#         joint2 = skeletons[:, :, joint2_idx, :]  # [batch, frames, coordinates]
        
#         # Calculate bone lengths for each frame
#         bone_lengths = tf.sqrt(tf.reduce_sum(tf.square(joint1 - joint2), axis=-1))  # [batch, frames]
        
#         # Calculate variance of bone length across frames (should be near 0)
#         mean_lengths = tf.reduce_mean(bone_lengths, axis=1, keepdims=True)  # [batch, 1]
#         length_variance = tf.reduce_mean(tf.square(bone_lengths - mean_lengths))
        
#         losses.append(length_variance)
    
#     return tf.reduce_mean(losses)

# def motion_smoothness_loss(skeletons):
#     """
#     Penalize jerky, non-smooth motion between frames
#     """
#     # Calculate velocity (first derivative of position)
#     velocity = skeletons[:, 1:] - skeletons[:, :-1]  # [batch, frames-1, joints, coords]
    
#     # Calculate acceleration (second derivative of position)
#     acceleration = velocity[:, 1:] - velocity[:, :-1]  # [batch, frames-2, joints, coords]
    
#     # Penalize large accelerations (encourage smooth motion)
#     return tf.reduce_mean(tf.square(acceleration))

# def combined_generator_loss(fake_output, generated_skeletons, real_skeletons, joint_connections):
#     # Original generator loss (adversarial component)
#     adv_loss = -tf.reduce_mean(fake_output)
    
#     # Reconstruction loss
#     mse_loss = tf.reduce_mean(tf.square(generated_skeletons - real_skeletons))
    
#     # Bone length consistency
#     bone_loss = bone_length_consistency_loss(generated_skeletons, joint_connections)
    
#     # Motion smoothness
#     motion_loss = motion_smoothness_loss(generated_skeletons)
    
#     # Combine losses with weighting
#     return adv_loss + 10.0 * mse_loss + 5.0 * bone_loss + 2.0 * motion_loss