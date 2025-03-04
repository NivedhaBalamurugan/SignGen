import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import show_output
from config import *
from utils.data_utils import load_word_embeddings


NOISE_DIM = 50
MAX_FRAMES = 233
NUM_JOINTS = 49
NUM_COORDINATES = 3

generator = tf.keras.models.load_model("Dataset/cgan_generator.keras")
word_embeddings = load_word_embeddings(GLOVE_TXT_PATH)


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

def get_cgan_sequence(word, isSave_Video):

    generated_sequence = generate_skeleton_sequence(word)
    if generated_sequence is not None:
        print(f"Generated Skeleton Shape for '{word}': {generated_sequence.shape}")
        save_path = "Dataset"
        os.makedirs(save_path, exist_ok=True)
        json_filepath = os.path.join(save_path, f"generated_skeleton_{word}.json")
        with open(json_filepath, 'w') as json_file:
            json.dump(generated_sequence.tolist(), json_file)
            print("Saved successfully")
        if isSave_Video:
            show_output.save_generated_sequence(generated_sequence, CGAN_OUTPUT_FRAMES, CGAN_OUTPUT_VIDEO) 
    return generated_sequence

if __name__ == "__main__":
    get_cgan_sequence("afternoon",1)