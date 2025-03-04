import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, Input, RepeatVector, Reshape
from config import *

def build_generator():
    inputs = tf.keras.Input(shape=(CGAN_NOISE_DIM + 50,))  # Noise + word embeddings
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(MAX_FRAMES * NUM_JOINTS * 3, activation="tanh")(x)  # Output skeleton
    outputs = tf.reshape(outputs, (-1, MAX_FRAMES, NUM_JOINTS, 3))
    return tf.keras.Model(inputs, outputs)

def build_discriminator():
    input_shape = (MAX_FRAMES, NUM_JOINTS, 53)  # 3 (skeleton) + 50 (word embeddings)
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1)(x)  # No activation for Wasserstein Loss
    return tf.keras.Model(inputs, outputs)

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
