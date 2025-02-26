import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHUNK_INDEX = 0
FP_PRECISION = np.float32

VIDEOS_PATH = "Dataset/videos"
WLASL_JSON_PATH = "Dataset/WLASL_v0.3.json"
MISSING_TXT_PATH = "Dataset/missing.txt"
CHUNKS_JSON_PATH = "Dataset/chunks.json"

LANDMARKS_PATH = "Dataset/landmarks/"
AUGMENTED_PATH = f"{LANDMARKS_PATH}/augmented/"
PALM_BODY_PATH = f"{LANDMARKS_PATH}/palm_body/"
MERGED_PATH = f"{LANDMARKS_PATH}/merged/"
FINAL_PATH = f"{LANDMARKS_PATH}/final/"


PALM_JSONL_PATH = f"{PALM_BODY_PATH}{CHUNK_INDEX}_palm_landmarks.jsonl.gz"
BODY_JSONL_PATH = f"{PALM_BODY_PATH}{CHUNK_INDEX}_body_landmarks.jsonl.gz"
CHECKPOINT_PATH = f"{PALM_BODY_PATH}{CHUNK_INDEX}_checkpoint.json"

MERGED_JSONL_PATH = f"{MERGED_PATH}{CHUNK_INDEX}_landmarks.jsonl.gz"
AUGMENTED_JSONL_PATH = f"{AUGMENTED_PATH}{CHUNK_INDEX}_aug_landmarks.jsonl.gz"
FINAL_JSONL_PATH = f"{FINAL_PATH}{CHUNK_INDEX}_final_landmarks.jsonl.gz"

EXT_BODY_JSONL_PATH = f"{CHUNK_INDEX}_str_body_landmarks.jsonl"
EXT_PALM_JSONL_PATH = f"{CHUNK_INDEX}_str_palm_landmarks.jsonl"
EXT_MERGED_JSONL_PATH = f"{CHUNK_INDEX}_str_landmarks.jsonl"
EXT_AUGMENTED_JSONL_PATH = f"{CHUNK_INDEX}_str_aug_landmarks.jsonl"





