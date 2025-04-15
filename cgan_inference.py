import tensorflow as tf
import numpy as np
import show_output
from config import *
from utils.data_utils import load_word_embeddings

INPUT_WORDS = ["friend","movie","time","fine","book", "now", "help", "money", "hat", "flower"]
generator = tf.keras.models.load_model("Models/cgan_model/generator_epoch30_loss0.1639.keras")

def generate_skeleton_sequence(word):
    word = check_extended_words(word)
    word_embeddings = load_word_embeddings()
    if word not in word_embeddings:
        print(f"Word '{word}' not found in embeddings.")
        return None

    word_vector = np.array(word_embeddings[word], dtype=np.float32).reshape(1, -1)
    noise = tf.random.normal([1, CGAN_NOISE_DIM], dtype=tf.float32)
    generator_input = tf.concat([noise, word_vector], axis=1)
    
    generated_skeleton = generator(generator_input, training=False).numpy()
    return generated_skeleton.squeeze()

def get_cgan_sequence(word, isSave=True):
    generated_sequence = generate_skeleton_sequence(word)
    if generated_sequence is None:
        return None
        
    print(f"Generated sequence for '{word}': {generated_sequence.shape}")
    
    if isSave:
        show_output.save_generated_sequence(generated_sequence, CGAN_OUTPUT_FRAMES, CGAN_OUTPUT_VIDEO)
        
    return generated_sequence

if __name__ == "__main__":
    for word in INPUT_WORDS:
        get_cgan_sequence(word)