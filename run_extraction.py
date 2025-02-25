from utils.jsonl_utils import extract_jsonl_gz 

CHUNK_INDEX = 0

BODY_JSONL_PATH = f"{CHUNK_INDEX}_body_landmarks.jsonl.gz"
PALM_JSONL_PATH = f"{CHUNK_INDEX}_palm_landmarks.jsonl.gz"
MERGED_JSONL_PATH = f"./landmarks/augmented/{CHUNK_INDEX}_aug_landmarks.jsonl.gz"

EXT_BODY_JSONL_PATH = f"{CHUNK_INDEX}_str_body_landmarks.jsonl"
EXT_PALM_JSONL_PATH = f"{CHUNK_INDEX}_str_palm_landmarks.jsonl"
EXT_MERGED_JSONL_PATH = f"{CHUNK_INDEX}_aug_landmarks.jsonl"

# extract_jsonl_gz(BODY_JSONL_PATH, EXT_BODY_JSONL_PATH)
# extract_jsonl_gz(PALM_JSONL_PATH, EXT_BODY_JSONL_PATH)
extract_jsonl_gz(MERGED_JSONL_PATH, EXT_MERGED_JSONL_PATH)
