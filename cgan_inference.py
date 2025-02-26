import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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

generator = tf.keras.models.load_model("Dataset/Gen.keras")
word_embeddings = load_word_embeddings("glove.6B.50d.txt")


def generate_skeleton_sequence(word):
    if word not in word_embeddings:
        print(f"Word '{word}' not found in embeddings.")
        return None

    word_vector = np.array(
        word_embeddings[word], dtype=np.float32).reshape(1, -1)
    noise = tf.random.normal([1, NOISE_DIM], dtype=tf.float32)
    generator_input = tf.concat([word_vector, noise], axis=1)

    generated_skeleton = generator(generator_input, training=False).numpy()
    return generated_skeleton.squeeze()


def visualize_skeleton(sequence):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for frame in range(0, sequence.shape[0], max(1, sequence.shape[0] // 10)):
        ax.clear()
        joints = sequence[frame]
        x, y, z = joints[:, 0], joints[:, 1], joints[:, 2]
        ax.scatter(x, y, z, c='r', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame}')
        plt.pause(0.1)

    plt.show()


word = "hello"  # needs to be changed based on user's input later
generated_sequence = generate_skeleton_sequence(word)
if generated_sequence is not None:
    print(f"Generated Skeleton Shape for '{word}': {generated_sequence.shape}")
    save_path = "Dataset"
    os.makedirs(save_path, exist_ok=True)
    json_filepath = os.path.join(save_path, f"generated_skeleton_{word}.json")
    with open(json_filepath, 'w') as json_file:
        json.dump(generated_sequence.tolist(), json_file)
        print("Saved successfully")
    visualize_skeleton(generated_sequence)
