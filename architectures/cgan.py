import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.layers import GRU, Dense, Input, Flatten, Dropout, BatchNormalization, Reshape, RepeatVector
from config import *



def build_generator():
    inputs = Input(shape=(CGAN_NOISE_DIM + 50,))  # Noise + word embeddings
    x = Dense(256, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Expand latent vector to a sequence of MAX_FRAMES time steps
    x = RepeatVector(MAX_FRAMES)(x)  # Now shape is (batch_size, MAX_FRAMES, 256)
    
    # Apply GRU layers to model temporal dependencies
    x = GRU(128, return_sequences=True)(x)
    x = GRU(128, return_sequences=True)(x)
    
    # Map to the output dimension: NUM_JOINTS * 3 per frame
    x = Dense(NUM_JOINTS * 2, activation="tanh")(x)
    outputs = Reshape((MAX_FRAMES, NUM_JOINTS, 2))(x)
    return Model(inputs, outputs)

def build_discriminator():
    input_shape = (MAX_FRAMES, NUM_JOINTS, 52)
    inputs = Input(shape=input_shape)
    
    # Flatten the input
    x = Flatten()(inputs)
    
    # Add dense layers
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(1)(x)  # No activation for Wasserstein Loss
    
    return Model(inputs, outputs)

def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    """Calculate the discriminator loss for Wasserstein GAN."""
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)  
