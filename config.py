import os
import numpy as np
import logging
from logging.handlers import RotatingFileHandler

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
CVAE_OUTPUT_FRAMES = os.path.join(OUTPUTS_PATH, "cvae_frames1")
CGAN_OUTPUT_FRAMES = os.path.join(OUTPUTS_PATH, "cgan_frames")
CVAE_STGCN_OUTPUT_FRAMES = os.path.join(OUTPUTS_PATH, "cvae_stgcn_frames")
CGAN_STGCN_OUTPUT_FRAMES = os.path.join(OUTPUTS_PATH, "cgan_stgcn_frames")
MHA_OUTPUT_FRAMES = os.path.join(OUTPUTS_PATH, "mha_frames")


#Final output Video paths
CVAE_OUTPUT_VIDEO = os.path.join(OUTPUTS_PATH, "cvae_video", "video.mp4")
CGAN_OUTPUT_VIDEO = os.path.join(OUTPUTS_PATH, "cgan_video", "video.mp4")
CVAE__STGCN_OUTPUT_VIDEO = os.path.join(OUTPUTS_PATH, "cvae_stgcn_video", "video.mp4")
CGAN_STGCN_OUTPUT_VIDEO = os.path.join(OUTPUTS_PATH, "cgan_stgcn_video", "video.mp4")
MHA_OUTPUT_VIDEO = os.path.join(OUTPUTS_PATH, "mha_video", "video.mp4")

FRAME_WIDTH = 224  
FRAME_HEIGHT = 224 

def get_paths(chunk_index=0):
    return {
        'palm_jsonl': os.path.join(PALM_BODY_PATH, f"{chunk_index}_palm_landmarks.jsonl.gz"),
        'body_jsonl': os.path.join(PALM_BODY_PATH, f"{chunk_index}_body_landmarks.jsonl.gz"),
        'checkpoint': os.path.join(PALM_BODY_PATH, f"{chunk_index}_checkpoint.json"),
        'merged_jsonl': os.path.join(MERGED_PATH, f"{chunk_index}_landmarks.jsonl.gz"),
        'augmented_jsonl': os.path.join(AUGMENTED_PATH, f"{chunk_index}_aug_landmarks.jsonl.gz"),
        'final_jsonl': os.path.join(FINAL_PATH, f"{chunk_index}_final_landmarks.jsonl.gz"),
        'ext_body': f"{chunk_index}_str_body_landmarks.jsonl",
        'ext_palm': f"{chunk_index}_str_palm_landmarks.jsonl",
        'ext_merged': f"{chunk_index}_str_landmarks.jsonl",
        'ext_augmented': f"{chunk_index}_str_aug_landmarks.jsonl"
    }

# Final landmarks paths
FINAL_JSONL_PATHS = os.path.join(FINAL_PATH, "*.jsonl")

# Model paths
CGAN_MODEL_PATH = os.path.join(MODELS_PATH, "cgan_model")
CGAN_GEN_PATH = os.path.join(CGAN_MODEL_PATH, "cgan_generator.keras")
CGAN_DIS_PATH = os.path.join(CGAN_MODEL_PATH, "cgan_discriminator.keras")
CVAE_MODEL_PATH = os.path.join(MODELS_PATH, "cvae_model")
STGCN_MODEL_PATH = os.path.join(MODELS_PATH, "stgcn_model")

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
CVAE_HIDDEN_DIM = 256
CVAE_LATENT_DIM = 32
CVAE_BATCH_SIZE = 64

#STGAN parameters
IN_CHANNELS = 3
NUM_NODES = 49
HIDDEN_DIM = 128

# Global variable to store embeddings
_WORD_EMBEDDINGS = None

def load_word_embeddings(filepath):
    """Lazy loading of word embeddings using singleton pattern"""
    global _WORD_EMBEDDINGS
    
    if _WORD_EMBEDDINGS is not None:
        return _WORD_EMBEDDINGS
        
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
        _WORD_EMBEDDINGS = word_embeddings
        return _WORD_EMBEDDINGS
    except Exception as e:
        logging.error(f"Error loading word embeddings: {e}")
        return None

#Glove parameters
EMBEDDING_DIM = 50

def get_word_embeddings():
    return load_word_embeddings(GLOVE_TXT_PATH)

WORD_EMBEDDINGS = get_word_embeddings()


def setup_logging(model):
    from datetime import datetime
    
    log_dir = os.path.join(BASE_PATH, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Add timestamp to log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model}_{timestamp}.log")
    
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.handlers = []
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"Logging to: {log_file}")