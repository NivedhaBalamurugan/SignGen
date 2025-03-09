import os
import json
import tensorflow as tf
import numpy as np
import show_output
from config import *
from utils.data_utils import load_word_embeddings

def generate_skeleton_sequence(word, generator):
    word_embeddings = load_word_embeddings()
    if word not in word_embeddings:
        word=WORD_NOT_FOUND
    word_vector = np.array(word_embeddings[word], dtype=np.float32).reshape(1, -1)
    noise = tf.random.normal([1, CGAN_NOISE_DIM], dtype=tf.float32)
    generator_input = tf.concat([noise, word_vector], axis=1)
    
    generated_skeleton = generator(generator_input, training=False).numpy()
    return generated_skeleton.squeeze()

def get_cgan_sequence(word, isSave_Video, model_path):
    generator = tf.keras.models.load_model(model_path)
    generated_sequence = generate_skeleton_sequence(word, generator)
    if generated_sequence is not None:
        print(f"Generated Skeleton Shape for '{word}': {generated_sequence.shape}")
        if isSave_Video:
            show_output.save_generated_sequence(generated_sequence, CGAN_OUTPUT_FRAMES, CGAN_OUTPUT_VIDEO) 
    return generated_sequence


def main_c(INPUT_WORD, isSave):
    return get_cgan_sequence(INPUT_WORD,isSave, get_cgan_path(INPUT_WORD))

if __name__ == "__main__":
    INPUT_WORD = "before"
    main_c(INPUT_WORD,1)