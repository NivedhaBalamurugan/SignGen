import os
import json
import tensorflow as tf
import numpy as np
import show_output
from config import *
from utils.data_utils import load_word_embeddings

INPUT_WORD = "before"

generator = tf.keras.models.load_model("Models\cgan_model\checkpoints\generator_epoch1_loss763.3392.keras")


def generate_skeleton_sequence(word):
    word_embeddings = load_word_embeddings(GLOVE_TXT_PATH)
    if word not in word_embeddings:
        print(f"Word '{word}' not found in embeddings.")
        return None

    word_vector = np.array(word_embeddings[word], dtype=np.float32).reshape(1, -1)
    # Generate noise with correct dimension (CGAN_NOISE_DIM)
    noise = tf.random.normal([1, CGAN_NOISE_DIM], dtype=tf.float32)
    generator_input = tf.concat([noise, word_vector], axis=1)
    
    generated_skeleton = generator(generator_input, training=False).numpy()
    return generated_skeleton.squeeze()

def get_cgan_sequence(word, isSave_Video):
    generated_sequence = generate_skeleton_sequence(word)
    if generated_sequence is not None:
        print(f"Generated Skeleton Shape for '{word}': {generated_sequence.shape}")
        save_path = OUTPUTS_PATH
        os.makedirs(save_path, exist_ok=True)
        json_filepath = os.path.join(save_path, f"generated_skeleton_{word}.json")
        with open(json_filepath, 'w') as json_file:
            json.dump(generated_sequence.tolist(), json_file)
            print("Saved successfully")
        if isSave_Video:
            show_output.save_generated_sequence(generated_sequence, CGAN_OUTPUT_FRAMES, CGAN_OUTPUT_VIDEO) 
    return generated_sequence

if __name__ == "__main__":
    get_cgan_sequence(INPUT_WORD,1)