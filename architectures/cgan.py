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

def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)  # Wasserstein Loss