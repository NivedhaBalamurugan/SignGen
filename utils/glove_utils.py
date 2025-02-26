import os
import numpy as np
from config import *

def load_word_embeddings(filepath):
    if not os.path.exists(filepath):
        logging.error(f"Word embeddings file not found: {filepath}")
        return None
        
    word_embeddings = {}
    try:
        with open(filepath, encoding="utf8") as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype=FP_PRECISION)
                word_embeddings[word] = vector
        logging.info(f"Loaded {len(word_embeddings)} word embeddings")
        return word_embeddings
    except Exception as e:
        logging.error(f"Error loading word embeddings: {e}")
        return None

def validate_word_embeddings(word_embeddings, expected_dim=50):
    if not word_embeddings:
        return False
        
    sample_vector = next(iter(word_embeddings.values()))
    if sample_vector.shape[0] != expected_dim:
        logging.error(f"Invalid embedding dimension: {sample_vector.shape[0]}, expected: {expected_dim}")
        return False
        
    return True