import os
import gzip
import orjson
from collections import defaultdict

CHUNK_INDEX = 0

BODY_JSONL_PATH = f"{CHUNK_INDEX}_body_landmarks.jsonl.gz"
PALM_JSONL_PATH = f"{CHUNK_INDEX}_palm_landmarks.jsonl.gz"
MERGED_JSONL_PATH = f"{CHUNK_INDEX}_landmarks.jsonl.gz"

def load_jsonl_gz(file_path):
    data = defaultdict(list)
    if not os.path.exists(file_path):
        return data
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            item = orjson.loads(line)
            data.update(item)
    return data

def merge_landmarks(body_data, palm_data):
    merged_data = defaultdict(list)
    for word in set(body_data.keys()).union(palm_data.keys()):
        for body_video, palm_video in zip(body_data[word], palm_data[word]):
            merged_video = []
            for body_frame, palm_frame in zip(body_video, palm_video):
                merged_frame = body_frame + palm_frame
                merged_video.append(merged_frame)
            merged_data[word].append(merged_video)
    return merged_data

def save_jsonl_gz(file_path, data):
    with gzip.open(file_path, "wb") as f:
        for key, value in data.items():
            f.write(orjson.dumps({key: value}) + b"\n")

body_data = load_jsonl_gz(BODY_JSONL_PATH)
palm_data = load_jsonl_gz(PALM_JSONL_PATH)

merged_data = merge_landmarks(body_data, palm_data)

save_jsonl_gz(MERGED_JSONL_PATH, merged_data)

print(f"Merged landmarks saved to {MERGED_JSONL_PATH}.")
