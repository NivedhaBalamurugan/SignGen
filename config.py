import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "Dataset")
MODELS_PATH = os.path.join(BASE_PATH, "Models")
OUTPUTS_PATH = os.path.join(BASE_PATH, "Outputs")

FP_PRECISION = np.float32

# Dataset paths
VIDEOS_PATH = os.path.join(DATASET_PATH, "videos")
WLASL_JSON_PATH = os.path.join(DATASET_PATH, "WLASL_v0.3.json")
MISSING_TXT_PATH = os.path.join(DATASET_PATH, "missing.txt")
CHUNKS_JSON_PATH = os.path.join(DATASET_PATH, "chunks.json")
GLOVE_TXT_PATH = os.path.join(DATASET_PATH, "glove", "glove.6B.50d.txt")

# Landmarks paths
LANDMARKS_PATH = os.path.join(DATASET_PATH, "landmarks")
AUGMENTED_PATH = os.path.join(LANDMARKS_PATH, "augmented")
PALM_BODY_PATH = os.path.join(LANDMARKS_PATH, "palm_body")
MERGED_PATH = os.path.join(LANDMARKS_PATH, "merged")
FINAL_PATH = os.path.join(LANDMARKS_PATH, "final")

#Final output frame paths
CVAE_OUTPUT_FRAMES = os.path.join(OUTPUTS_PATH, "cvae_frames")
CGAN_OUTPUT_FRAMES = os.path.join(OUTPUTS_PATH, "cgan_frames")

#Final output Video paths
CVAE_OUTPUT_VIDEO = os.path.join(OUTPUTS_PATH, "cvae_video", "video.mp4")
CGAN_OUTPUT_VIDEO = os.path.join(OUTPUTS_PATH, "cgan_video", "video.mp4")


def get_paths(chunk_index=0):
    return {
        'palm_jsonl': os.path.join(PALM_BODY_PATH, f"{chunk_index}_palm_landmarks.jsonl.gz"),
        'body_jsonl': os.path.join(PALM_BODY_PATH, f"{chunk_index}_body_landmarks.jsonl.gz"),
        'checkpoint': os.path.join(PALM_BODY_PATH, f"{chunk_index}_checkpoint.json"),
        'merged_jsonl': os.path.join(MERGED_PATH, f"{chunk_index}_landmarks.jsonl.gz"),
        'augmented_jsonl': os.path.join(AUGMENTED_PATH, f"{chunk_index}_aug_landmarks.jsonl.gz"),
        'final_jsonl': os.path.join(FINAL_PATH, f"{chunk_index}_final_landmarks.jsonl.gz"),
        'ext_body': f"{chunk_index}_str_body_landmarks.jsonl.gz",
        'ext_palm': f"{chunk_index}_str_palm_landmarks.jsonl.gz",
        'ext_merged': f"{chunk_index}_str_landmarks.jsonl.gz",
        'ext_augmented': f"{chunk_index}_str_aug_landmarks.jsonl.gz"
    }

# Final landmarks paths
FINAL_JSONL_PATHS = os.path.join(FINAL_PATH, "*.jsonl")

# Model paths
CGAN_GEN_PATH = os.path.join(MODELS_PATH, "cgan_generator.keras")
CGAN_DIS_PATH = os.path.join(MODELS_PATH, "cgan_discriminator.keras")
CGAN_TRAIN_HISTORY_PATH = os.path.join(MODELS_PATH, "cgan_train_history.npy")
CVAE_MODEL_PATH = os.path.join(MODELS_PATH, "cvae_model")

# Model parameters
MAX_FRAMES = 233
NUM_JOINTS = 49
NUM_COORDINATES = 3

# CGAN parameters
CGAN_BATCH_SIZE = 32
CGAN_EPOCHS = 100
CGAN_LEARNING_RATE = 1e-4
CGAN_LOG_INTERVAL = 10
CGAN_NOISE_DIM = 50
CGAN_MAX_GEN_LOSS = 10.0
CGAN_MAX_DISC_LOSS = 10.0

#CVAE parameters
CVAE_INPUT_DIM = 147  
CVAE_HIDDEN_DIM = 128
CVAE_LATENT_DIM = 64
CVAE_BATCH_SIZE = 16

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

#Glove parameters
EMBEDDING_DIM = 50
WORD_EMBEDDINGS = load_word_embeddings(GLOVE_TXT_PATH)

