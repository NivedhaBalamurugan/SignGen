from config import *

def validate_data_shapes(word_vectors, skeleton_sequences):
    if word_vectors.shape[0] != skeleton_sequences.shape[0]:
        logging.error(f"Mismatch in samples: words={word_vectors.shape[0]}, skeletons={skeleton_sequences.shape[0]}")
        return False
    
    if skeleton_sequences.shape[1:] != (MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES):
        logging.error(f"Invalid skeleton shape: {skeleton_sequences.shape[1:]}, expected: ({MAX_FRAMES}, {NUM_JOINTS}, {NUM_COORDINATES})")
        return False
    
    return True

def validate_config():
    if MAX_FRAMES <= 0:
        logging.error("MAX_FRAMES must be positive")
        return False
    if CGAN_BATCH_SIZE <= 0:
        logging.error("BATCH_SIZE must be positive")
        return False
    if CGAN_EPOCHS <= 0:
        logging.error("EPOCHS must be positive")
        return False
    if CGAN_LEARNING_RATE <= 0:
        logging.error("LEARNING_RATE must be positive")
        return False
    return True