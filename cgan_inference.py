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

   
    real_normalized = (real_sequence - np.mean(real_sequence, axis=0)) / (np.std(real_sequence, axis=0) + 1e-10)
    gen_normalized = (generated_sequence - np.mean(generated_sequence, axis=0)) / (np.std(generated_sequence, axis=0) + 1e-10)
    
   
    real_normalized = np.nan_to_num(real_normalized)
    gen_normalized = np.nan_to_num(gen_normalized)

 
    mse = np.mean((real_normalized - gen_normalized) ** 2)
    mse_score = 1.0 / (1.0 + mse)  
    
    
    try:
        real_flat = real_normalized.reshape(real_normalized.shape[0], -1)
        gen_flat = gen_normalized.reshape(gen_normalized.shape[0], -1)
        
 
        correlation_matrix = np.corrcoef(real_flat.flatten(), gen_flat.flatten())
        corr_score = (correlation_matrix[0, 1] + 1) / 2  # Map from [-1,1] to [0,1]
    except:
        corr_score = 0.0  

 
    real_fft = np.abs(np.fft.fft(real_normalized, axis=0))
    gen_fft = np.abs(np.fft.fft(gen_normalized, axis=0))
    
 
    real_fft_norm = real_fft / (np.sum(real_fft) + 1e-10)
    gen_fft_norm = gen_fft / (np.sum(gen_fft) + 1e-10)
    
   
    fft_diff = np.mean(np.abs(real_fft_norm - gen_fft_norm))
    fft_score = 1.0 - np.minimum(1.0, fft_diff)
    

    final_score = 0.4 * mse_score + 0.3 * corr_score + 0.3 * fft_score
    

    final_score = max(0, min(1, final_score))
    
    
    
    return final_score


    

def get_diversity_score(input_word,seq):
   
    real_sequence = get_real_data(input_word)
    score = calculate_diversity_score(real_sequence, seq)
    
    return score

if __name__ == "__main__":
  seq = get_cgan_sequence(INPUT_WORD)
    





