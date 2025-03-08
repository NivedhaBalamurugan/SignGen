import os
import numpy as np
from config import *

def validate_word_embeddings(word_embeddings, expected_dim=EMBEDDING_DIM):
    if not word_embeddings:
        return False
        
    sample_vector = next(iter(word_embeddings.values()))
    if sample_vector.shape[0] != expected_dim:
        logging.error(f"Invalid embedding dimension: {sample_vector.shape[0]}, expected: {expected_dim}")
        return False
        
    return True
