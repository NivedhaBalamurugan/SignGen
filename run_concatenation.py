import os
from utils.concat_landmarks import merge_landmarks
from config import *

def merge_chunk(index):
    if not os.path.exists(PALM_BODY_PATH):
        logging.error(f"Path does not exist: {PALM_BODY_PATH}")
        return

    os.makedirs(MERGED_PATH, exist_ok=True)
    paths = get_paths(index)
    body_json_path = paths['body_jsonl']
    palm_json_path = paths['palm_jsonl']
    merged_json_path = paths['merged_jsonl']
    merge_landmarks(body_json_path, palm_json_path, merged_json_path)
    logging.info(f"Chunk {index} merged and saved to {merged_json_path}")

def merge_all_chunks(size=10):
    for i in range(size):
        merge_chunk(i)


