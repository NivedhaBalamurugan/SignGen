import tensorflow as tf
import numpy as np
from scipy import signal
from utils.data_utils import get_real_data, load_word_embeddings
from sklearn.decomposition import PCA
from config import *
import os
import json
import logging
import show_output

INPUT_WORD = "friend"

generator = tf.keras.models.load_model("Models/cgan_model/gen_model.keras")

def generate_skeleton_sequence(word, fixed_seed=True):
    word_embeddings = load_word_embeddings()
    word_vector = np.array(word_embeddings[word], dtype=np.float32).reshape(1, -1)

    if fixed_seed:

        word_hash = hash(word) % 10000
        tf.random.set_seed(word_hash)
        np.random.seed(word_hash)
    
    noise = tf.random.normal([1, CGAN_NOISE_DIM], dtype=tf.float32)
    generator_input = tf.concat([noise, word_vector], axis=1)
    
    generated_skeleton = generator(generator_input).numpy()
    return generated_skeleton.squeeze()

def get_cgan_sequence(word, isSave=True):
    word = check_extended_words(word.lower())
    generated_sequence = generate_skeleton_sequence(word)
    if generated_sequence is None:
        return None

    print(f"Generated CGAN sequence for '{word}': {generated_sequence.shape}")
    diversity_score = get_diversity_score(word,generated_sequence)
    print(f"Diversity score for '{word}': {diversity_score:.4f}")

    if isSave:
        show_output.save_generated_sequence(generated_sequence, CGAN_OUTPUT_FRAMES, CGAN_OUTPUT_VIDEO)

    return generated_sequence, diversity_score

def calculate_diversity_score(real_sequence, generated_sequence):

    if real_sequence.shape[0] != generated_sequence.shape[0]:
        if real_sequence.shape[0] < generated_sequence.shape[0]:
            indices = np.linspace(0, real_sequence.shape[0] - 1, generated_sequence.shape[0], dtype=int)
            real_sequence = real_sequence[indices]
        else:
            indices = np.linspace(0, generated_sequence.shape[0] - 1, real_sequence.shape[0], dtype=int)
            generated_sequence = generated_sequence[indices]

    real_range = np.max(real_sequence, axis=0) - np.min(real_sequence, axis=0)
    gen_range = np.max(generated_sequence, axis=0) - np.min(generated_sequence, axis=0)
    range_diff = np.mean(np.abs(real_range - gen_range) / (np.mean(real_range) + 1e-10))
    range_score = np.tanh(range_diff * 3)
    
    real_speed = np.mean(np.abs(np.diff(real_sequence, axis=0)), axis=0)
    gen_speed = np.mean(np.abs(np.diff(generated_sequence, axis=0)), axis=0)
    speed_diff = np.mean(np.abs(real_speed - gen_speed) / (np.mean(real_speed) + 1e-10))
    speed_score = np.tanh(speed_diff * 3)   

    real_flat = real_sequence.reshape(real_sequence.shape[0], -1)
    gen_flat = generated_sequence.reshape(generated_sequence.shape[0], -1)
    
    
    pca_real = PCA(n_components=min(5, real_flat.shape[0], real_flat.shape[1]))
    pca_gen = PCA(n_components=min(5, gen_flat.shape[0], gen_flat.shape[1]))
    
    pca_real.fit(real_flat)
    pca_gen.fit(gen_flat)
    

    real_var = pca_real.explained_variance_ratio_
    gen_var = pca_gen.explained_variance_ratio_
    
    var_diff = np.mean(np.abs(real_var - gen_var))
    pca_score = np.tanh(var_diff * 10)

    real_psd = np.mean([np.mean(signal.welch(real_sequence[:, i])[1]) for i in range(real_sequence.shape[1])])
    gen_psd = np.mean([np.mean(signal.welch(generated_sequence[:, i])[1]) for i in range(generated_sequence.shape[1])])
    
    psd_diff = np.abs(real_psd - gen_psd) / (real_psd + 1e-10)
    psd_score = np.tanh(psd_diff * 5)
    
    final_score = 0.25 * range_score + 0.25 * speed_score + 0.25 * pca_score + 0.25 * psd_score
    
    final_score = max(0, min(1, final_score))
    
    return final_score

def get_diversity_score(input_word, generated_sequence):
    real_sequence = get_real_data(input_word)    
    
    score = calculate_diversity_score(real_sequence, generated_sequence)
    
    return score

if __name__ == "__main__":
  seq = get_cgan_sequence(INPUT_WORD)
    





