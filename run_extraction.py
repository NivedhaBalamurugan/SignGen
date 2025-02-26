import os
from utils.jsonl_utils import extract_jsonl_gz
from config import *

def extract_body_palm_chunk(index):
    paths = get_paths(index)
    if not os.path.exists(PALM_BODY_PATH):
        logging.error(f"Path does not exist: {PALM_BODY_PATH}")
        return

    body_json_path = paths['body_jsonl']
    palm_json_path = paths['palm_jsonl']

    ext_body_json_path = paths['ext_body']
    ext_palm_json_path = paths['ext_palm']

    extract_jsonl_gz(body_json_path, ext_body_json_path)
    extract_jsonl_gz(palm_json_path, ext_palm_json_path)
    logging.info(f"Chunk {index} body and palm landmarks extracted to {ext_body_json_path} and {ext_palm_json_path}")

def extract_merged_chunk(index):
    paths = get_paths(index)
    if not os.path.exists(MERGED_PATH):
        logging.error(f"Path does not exist: {MERGED_PATH}")
        return

    merged_json_path = paths['merged_jsonl']
    ext_merged_json_path = paths['ext_merged']

    extract_jsonl_gz(merged_json_path, ext_merged_json_path)
    logging.info(f"Chunk {index} merged landmarks extracted to {ext_merged_json_path}")

def extract_augmented_chunk(index):
    paths = get_paths(index)
    if not os.path.exists(AUGMENTED_PATH):
        logging.error(f"Path does not exist: {AUGMENTED_PATH}")
        return

    augmented_json_path = paths['augmented_jsonl']
    ext_augmented_json_path = paths['ext_augmented']

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
