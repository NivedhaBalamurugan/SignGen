import os
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import json

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
ONE_HOT_TXT_PATH = os.path.join(DATASET_PATH, "0_landmarks_top50_aug100_stats.json")
EXTENDED_WORD_PATH = os.path.join(DATASET_PATH, "extended_words.json")
ALPHABET_PATH = os.path.join(DATASET_PATH, "alphabets")

# Landmarks paths
LANDMARKS_PATH = os.path.join(DATASET_PATH, "landmarks")
AUGMENTED_PATH = os.path.join(LANDMARKS_PATH, "augmented")
PALM_BODY_PATH = os.path.join(LANDMARKS_PATH, "palm_body")
MERGED_PATH = os.path.join(LANDMARKS_PATH, "merged")
FINAL_PATH = os.path.join(LANDMARKS_PATH, "final")

#Final output frame paths
CVAE_OUTPUT_FRAMES = os.path.join(OUTPUTS_PATH, "cvae_frames")
CGAN_OUTPUT_FRAMES = os.path.join(OUTPUTS_PATH, "cgan_frames")
MHA_OUTPUT_FRAMES = os.path.join(OUTPUTS_PATH, "mha_frames")

#Final output Video paths
CVAE_OUTPUT_VIDEO = os.path.join(OUTPUTS_PATH, "cvae_video", "video.mp4")
CGAN_OUTPUT_VIDEO = os.path.join(OUTPUTS_PATH, "cgan_video", "video.mp4")
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
FINAL_JSONL_GZ_PATHS = os.path.join(FINAL_PATH, "*.jsonl.gz")

# Model paths
CGAN_MODEL_PATH = os.path.join(MODELS_PATH, "cgan_model", "checkpoints")
CGAN_DIS_PATH = os.path.join(CGAN_MODEL_PATH, "cgan_discriminator.keras")
CVAE_MODEL_PATH = os.path.join(MODELS_PATH, "cvae_model")
STGCN_MODEL_PATH = os.path.join(MODELS_PATH, "stgcn_model")

# Model parameters
MAX_FRAMES = 30
NUM_LINE_SEG = 27
NUM_JOINTS = 29
NUM_COORDINATES = 2
NOSE_VALUE = [0.50327, 0.26939]
RIGHT_HIP_VALUE =  [0.58529, 0.92522]
LEFT_HIP_VALUE = [0.42915, 0.93126]
RIGHT_SHOULDER_VALUE = [0.63497, 0.48466]
LEFT_SHOULDER_VALUE =  [0.37355, 0.48285]
RIGHT_ELBOW_VALUE = [0.65479696, 0.7519779 ]
LEFT_ELBOW_VALUE = [0.30541024, 0.75331193]
ISLETTER = False

#CVAE parameters
CVAE_BATCH_SIZE = 100
learning_rate = 1e-3
max_epoch = 200
latent_size = 64
learning_rate = 1e-4
batch_size = 64

#Glove parameters
EMBEDDING_DIM = 50

# CGAN parameters
CGAN_BATCH_SIZE = 64
CGAN_EPOCHS = 100
CGAN_LEARNING_RATE = 1e-4
CGAN_LOG_INTERVAL = 10
CGAN_NOISE_DIM = 50

#STGAN parameters
IN_CHANNELS = 3
NUM_NODES = 29
HIDDEN_DIM = 128

def setup_logging(model):
    from datetime import datetime
    
    log_dir = os.path.join(BASE_PATH, "Logs")
    os.makedirs(log_dir, exist_ok=True)
    
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


def get_cgan_path(word):
    word_to_number = {"before": 12, "again": 9, "balance": 6, "argue": 10, "already": 11, "awkward": 11}
    if word not in word_to_number:
        number = 12 
    else:
        number = word_to_number[word]
    CGAN_GEN_PATH = os.path.join(CGAN_MODEL_PATH, f"generator_epoch{number}.keras")
    return CGAN_GEN_PATH

def check_extended_words(actual_gloss):

    try:
        with open(EXTENDED_WORD_PATH, 'r') as file:
            synonyms_dictionary = json.load(file)
    
        normalized_input = actual_gloss.lower().strip()
        
        if normalized_input in synonyms_dictionary:
            return normalized_input

        for key, synonyms in synonyms_dictionary.items():
            normalized_synonyms = [syn.lower().strip() for syn in synonyms]
            if normalized_input in normalized_synonyms:
                return key
        return actual_gloss 
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return actual_gloss


def get_main_words():
    main_words = []
    try:
        with open(EXTENDED_WORD_PATH, 'r') as file:
            synonyms_dictionary = json.load(file)
    
        for key, synonyms in synonyms_dictionary.items():
            main_words.append(key)

    except Exception as e:
        print(f"Error: {str(e)}")

    return main_words