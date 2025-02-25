from utils.concat_landmarks import merge_landmarks

CHUNK_INDEX = 0

BODY_JSONL_PATH = f"{CHUNK_INDEX}_body_landmarks.jsonl.gz"
PALM_JSONL_PATH = f"{CHUNK_INDEX}_palm_landmarks.jsonl.gz"
MERGED_JSONL_PATH = f"{CHUNK_INDEX}_landmarks.jsonl.gz"

merge_landmarks(BODY_JSONL_PATH, PALM_JSONL_PATH, MERGED_JSONL_PATH)
