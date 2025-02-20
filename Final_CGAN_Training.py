import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input, RepeatVector


def load_word_embeddings(filepath):
    word_embeddings = {}
    with open(filepath, 'r') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = vector
    return word_embeddings


NOISE_DIM = 50


def load_skeleton_sequences(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def build_generator():
    model = Sequential([
        Input(shape=(50 + NOISE_DIM,)),
        RepeatVector(233),
        GRU(128, return_sequences=True),
        GRU(64, return_sequences=True),
        Dense(147, activation='linear')
    ])
    return model


def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(233, 147)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


# This dataset is available in kaggle
word_embeddings = load_word_embeddings("glove.6B.50d.txt")
# To be replaced with the json files of chunks
skeleton_data = load_skeleton_sequences("body_skeleton_data.json")
words = list(skeleton_data.keys())

# The below function might not be needed .. still run with it


def pad_sequences_3d(sequences, max_frames, num_joints, num_coordinates):

    padded = np.zeros((len(sequences), max_frames,
                      num_joints, num_coordinates))
    for i, seq in enumerate(sequences):
        seq = np.array(seq)[:max_frames]
        padded[i, :seq.shape[0], :, :] = seq
    return padded


words = [word for word in skeleton_data.keys() if word in word_embeddings]
word_vectors = np.array([word_embeddings[word] for word in words])
print(word_vectors.shape)

MAX_FRAMES = 233
NUM_JOINTS = 49
NUM_COORDINATES = 3

skeleton_sequences = pad_sequences_3d(
    [skeleton_data[word] for word in words],
    MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES
)

print(f"Word Vectors Shape: {word_vectors.shape}")
print(f"Skeleton Sequences Shape: {skeleton_sequences.shape}")

EMBEDDING_DIM = len(word_vectors[0])
HIDDEN_UNITS = 256
OUTPUT_DIM = skeleton_sequences.shape[1]
BATCH_SIZE = 32
EPOCHS = 100
generator = build_generator()
discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


print("Original word_vectors shape:", word_vectors.shape)


num_samples = word_vectors.shape[0]
num_batches = max(1, num_samples // BATCH_SIZE)
trimmed_size = num_batches * BATCH_SIZE


print(
    f"Total samples: {num_samples}, Batch size: {BATCH_SIZE}, Computed num_batches: {num_batches}")


if num_samples < BATCH_SIZE:
    print("Warning: Not enough samples to form a full batch!")
    num_batches = 1
    trimmed_size = num_samples


word_vectors = word_vectors[:trimmed_size]
word_vectors = word_vectors.reshape(trimmed_size, 50)


print("Processed word_vectors shape:", word_vectors.shape)


print("Original skeleton_sequences shape:", skeleton_sequences.shape)


skeleton_sequences = skeleton_sequences[:trimmed_size]


print("Processed skeleton_sequences shape:", skeleton_sequences.shape)


print(f"Final num_batches: {num_batches}")


def train_step(word_vector, real_skeleton):

    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    generator_input = tf.concat([word_vector, noise], axis=1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_skeleton = generator(generator_input, training=True)

        real_skeleton = tf.reshape(real_skeleton, (BATCH_SIZE, 233, -1))
        generated_skeleton = tf.reshape(
            generated_skeleton, (BATCH_SIZE, 233, -1))

        real_output = discriminator(real_skeleton, training=True)
        fake_output = discriminator(generated_skeleton, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


for epoch in range(EPOCHS):

    for i in range(num_batches):

        batch_indices = np.random.choice(
            trimmed_size, BATCH_SIZE, replace=False)

        word_batch = word_vectors[batch_indices]
        print(word_batch.shape)
        skeleton_batch = skeleton_sequences[batch_indices]
        skeleton_batch = skeleton_batch.reshape(BATCH_SIZE, 233, -1)

        try:

            gen_loss, disc_loss = train_step(word_batch, skeleton_batch)

            print(
                f"Batch {i+1}: Gen Loss = {gen_loss.numpy():.4f}, Disc Loss = {disc_loss.numpy():.4f}")

        except Exception as e:
            print(f"Error in train_step: {e}")
            exit()

    print(
        f"Epoch {epoch+1}/{EPOCHS} - Gen Loss: {gen_loss.numpy():.4f}, Disc Loss: {disc_loss.numpy():.4f}")

generator_path = "/Users/subramaniansenthilkumar/Desktop/FYP"  # To be replaced
discriminator_path = "/Users/subramaniansenthilkumar/Desktop/FYP"  # To be replaced


generator.save(generator_path)
discriminator.save(discriminator_path)
