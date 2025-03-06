import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.layers import GRU, Dense, Input, Flatten, Dropout, BatchNormalization, Reshape, RepeatVector
from config import *



def build_generator():
    inputs = Input(shape=(CGAN_NOISE_DIM + 50,))  # Noise + word embeddings
    x = Dense(128, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(MAX_FRAMES * NUM_JOINTS * 3, activation="tanh")(x)  # Output skeleton
    outputs = Reshape((MAX_FRAMES, NUM_JOINTS, 3))(outputs)  # FIXED Reshape
    return Model(inputs, outputs)

def build_discriminator():
    input_shape = (30, 49, 53)  # Fixed to 30 frames, 49 joints, and 53 features (3 + 50)
    inputs = tf.keras.Input(shape=input_shape)
    
    # Flatten the input
    x = tf.keras.layers.Flatten()(inputs)
    
    # Add dense layers
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(1)(x)  # No activation for Wasserstein Loss
    
    return tf.keras.Model(inputs, outputs)

def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    """Calculate the discriminator loss for Wasserstein GAN."""
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)  
