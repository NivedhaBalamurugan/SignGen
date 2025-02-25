import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input, RepeatVector, Reshape

def load_word_embeddings(filepath):
    word_embeddings = {}
    with open(filepath, encoding="utf8") as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = vector
    return word_embeddings

NOISE_DIM = 50
MAX_FRAMES = 233
NUM_JOINTS = 49
NUM_COORDINATES = 3

def load_skeleton_sequences(filepaths):
    skeleton_data = {}

    for filepath in filepaths:
        with open(filepath, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    word = list(data.keys())[0]
                    videos = data[word]
                    
                    if word not in skeleton_data:
                        skeleton_data[word] = []
                    
                  
                    skeleton_data[word].extend([pad_video(video, MAX_FRAMES) for video in videos])

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    return skeleton_data

def pad_video(video, max_frames):
    padded_video = np.zeros((max_frames, NUM_JOINTS, NUM_COORDINATES))
    video = np.array(video)[:max_frames] 
    padded_video[:video.shape[0], :, :] = video  
    return padded_video

def build_generator():
    model = Sequential([
        Input(shape=(50 + NOISE_DIM,)),
        RepeatVector(MAX_FRAMES),
        GRU(128, return_sequences=True),
        GRU(64, return_sequences=True),
        Dense(NUM_JOINTS * NUM_COORDINATES, activation='linear'),
        Reshape((MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES))
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES + 50)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


word_embeddings = load_word_embeddings("glove.6B.50d.txt")
jsonl_files = glob.glob("Dataset/*.jsonl")  # Change it 
skeleton_data = load_skeleton_sequences(jsonl_files)



words = [word for word in skeleton_data.keys() if word in word_embeddings]
word_vectors = np.array([word_embeddings[word] for word in words])


all_skeleton_sequences = []
all_word_vectors = []
for word in words:
    for video in skeleton_data[word]:  
        all_skeleton_sequences.append(video)
        all_word_vectors.append(word_embeddings[word])


all_skeleton_sequences = np.array(all_skeleton_sequences)
all_word_vectors = np.array(all_word_vectors)

print(f"Word Vectors Shape: {all_word_vectors.shape}")
print(f"Skeleton Sequences Shape: {all_skeleton_sequences.shape}")


EMBEDDING_DIM = all_word_vectors.shape[1]
BATCH_SIZE = 32
EPOCHS = 100


generator = build_generator()
discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

num_samples = all_word_vectors.shape[0]
num_batches = max(1, num_samples // BATCH_SIZE)
trimmed_size = num_batches * BATCH_SIZE


all_word_vectors = all_word_vectors[:trimmed_size]
all_skeleton_sequences = all_skeleton_sequences[:trimmed_size]

print(f"Final num_batches: {num_batches}")

def train_step(word_vector, real_skeleton):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM], dtype=tf.float32)  
    word_vector = tf.cast(word_vector, tf.float32)
    generator_input = tf.concat([word_vector, noise], axis=1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_skeleton = generator(generator_input, training=True)


        real_skeleton = tf.cast(real_skeleton, tf.float32)
        generated_skeleton = tf.cast(generated_skeleton, tf.float32)


        word_vector_expanded = tf.reshape(word_vector, (BATCH_SIZE, 1, 1, 50))  
        word_vector_expanded = tf.tile(word_vector_expanded, [1, MAX_FRAMES, NUM_JOINTS, 1])  

        real_input = tf.concat([real_skeleton, word_vector_expanded], axis=-1)  
        fake_input = tf.concat([generated_skeleton, word_vector_expanded], axis=-1)

        real_output = discriminator(real_input, training=True)
        fake_output = discriminator(fake_input, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss





for epoch in range(EPOCHS):
    for i in range(num_batches):
        batch_indices = np.random.choice(trimmed_size, BATCH_SIZE, replace=False)
        word_batch = all_word_vectors[batch_indices]
        skeleton_batch = all_skeleton_sequences[batch_indices]

        try:
            gen_loss, disc_loss = train_step(word_batch, skeleton_batch)
            print(f"Batch {i+1}: Gen Loss = {gen_loss.numpy():.4f}, Disc Loss = {disc_loss.numpy():.4f}")
        except Exception as e:
            print(f"Error in train_step: {e}")
            exit()

    print(f"Epoch {epoch+1}/{EPOCHS} - Gen Loss: {gen_loss.numpy():.4f}, Disc Loss = {disc_loss.numpy():.4f}")


generator.save("Dataset/Generator")
discriminator.save("Dataset/Discriminator")
