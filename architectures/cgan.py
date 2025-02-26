import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, Input, RepeatVector, Reshape
from config import *

def build_generator() -> tf.keras.Model:
    """Build and return the CGAN generator model.
    
    Returns:
        tf.keras.Model: A sequential model for generating skeleton sequences
    """
    model = Sequential([
        Input(shape=(50 + CGAN_NOISE_DIM,)),
        RepeatVector(MAX_FRAMES),
        GRU(128, return_sequences=True),
        GRU(64, return_sequences=True),
        Dense(NUM_JOINTS * NUM_COORDINATES, activation='linear'),
        Reshape((MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES))
    ])
    return model

def build_discriminator() -> tf.keras.Model:
    """Build and return the CGAN discriminator model.
    
    Returns:
        tf.keras.Model: A sequential model for discriminating real/fake sequences
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES + 50)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    """Calculate the discriminator loss.
    
    Args:
        real_output: Discriminator predictions on real data
        fake_output: Discriminator predictions on generated data
        
    Returns:
        tf.Tensor: Combined loss for real and fake predictions
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss