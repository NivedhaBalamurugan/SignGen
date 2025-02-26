import os
from utils.jsonl_utils import extract_jsonl_gz
from config import *

def extract_body_palm_chunk(index):
    if not os.path.exists(PALM_BODY_PATH):
        logging.error(f"Path does not exist: {PALM_BODY_PATH}")
        return

    body_json_path = BODY_JSONL_PATH.replace(f"{CHUNK_INDEX}", f"{index}")
    palm_json_path = PALM_JSONL_PATH.replace(f"{CHUNK_INDEX}", f"{index}")

    ext_body_json_path = EXT_BODY_JSONL_PATH.replace(f"{CHUNK_INDEX}", f"{index}")
    ext_palm_json_path = EXT_PALM_JSONL_PATH.replace(f"{CHUNK_INDEX}", f"{index}")

    extract_jsonl_gz(body_json_path, ext_body_json_path)
    extract_jsonl_gz(palm_json_path, ext_palm_json_path)
    logging.info(f"Chunk {index} body and palm landmarks extracted to {ext_body_json_path} and {ext_palm_json_path}")

def extract_merged_chunk(index):
    if not os.path.exists(MERGED_PATH):
        logging.error(f"Path does not exist: {MERGED_PATH}")
        return

    merged_json_path = MERGED_JSONL_PATH.replace(f"{CHUNK_INDEX}", f"{index}")
    ext_merged_json_path = EXT_MERGED_JSONL_PATH.replace(f"{CHUNK_INDEX}", f"{index}")

    extract_jsonl_gz(merged_json_path, ext_merged_json_path)
    logging.info(f"Chunk {index} merged landmarks extracted to {ext_merged_json_path}")

def extract_augmented_chunk(index):
    if not os.path.exists(AUGMENTED_PATH):
        logging.error(f"Path does not exist: {AUGMENTED_PATH}")
        return

    augmented_json_path = AUGMENTED_JSONL_PATH.replace(f"{CHUNK_INDEX}", f"{index}")
    ext_augmented_json_path = EXT_AUGMENTED_JSONL_PATH.replace(f"{CHUNK_INDEX}", f"{index}")

    extract_jsonl_gz(augmented_json_path, ext_augmented_json_path)
    logging.info(f"Chunk {index} augmented landmarks extracted to {ext_augmented_json_path}")

def extract_all_body_palm_chunks(size=10):
    for i in range(size):
        extract_body_palm_chunk(i)

def extract_all_merged_chunks(size=10):
    for i in range(size):
        extract_merged_chunk(i)

def extract_all_augmented_chunks(size=10):
    for i in range(size):
        extract_augmented_chunk(i)
