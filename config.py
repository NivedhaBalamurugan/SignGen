import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "Dataset")
MODELS_PATH = os.path.join(BASE_PATH, "Models")

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
FINAL_JSONL_PATHS = os.path.join(FINAL_PATH, "*.jsonl.gz")

# Model paths
CGAN_GEN_PATH = os.path.join(MODELS_PATH, "cgan_generator.keras")
CGAN_DIS_PATH = os.path.join(MODELS_PATH, "cgan_discriminator.keras")
CGAN_TRAIN_HISTORY_PATH = os.path.join(MODELS_PATH, "cgan_train_history.npy")

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