import os
from utils.concat_landmarks import merge_landmarks
from config import *

def merge_chunk(index):
    if not os.path.exists(PALM_BODY_PATH):
        logging.error(f"Path does not exist: {PALM_BODY_PATH}")
        return

    os.makedirs(MERGED_PATH, exist_ok=True)
    body_json_path = BODY_JSONL_PATH.replace(f"{CHUNK_INDEX}", f"{index}")
    palm_json_path = PALM_JSONL_PATH.replace(f"{CHUNK_INDEX}", f"{index}")
    merged_json_path = MERGED_JSONL_PATH.replace(f"{CHUNK_INDEX}", f"{index}")

    merge_landmarks(body_json_path, palm_json_path, merged_json_path)
    logging.info(f"Chunk {index} merged and saved to {merged_json_path}")

def merge_all_chunks(size=10):
    for i in range(size):
        merge_chunk(i)


