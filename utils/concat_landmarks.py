from collections import defaultdict
from utils.jsonl_utils import load_jsonl_gz, save_jsonl_gz

def merge_landmarks(body_landmarks_path, palm_landmarks_path, merged_landmarks_path):
    try:
        body_data = load_jsonl_gz(body_landmarks_path)
        palm_data = load_jsonl_gz(palm_landmarks_path)
    except Exception as e:
        print(f"Error loading JSONL files: {e}")
        return

    merged_data = defaultdict(list)
    try:
        for word in set(body_data.keys()).union(palm_data.keys()):
            for body_video, palm_video in zip(body_data[word], palm_data[word]):
                merged_video = []
                for body_frame, palm_frame in zip(body_video, palm_video):
                    merged_frame = body_frame + palm_frame
                    merged_video.append(merged_frame)
                merged_data[word].append(merged_video)
    except Exception as e:
        print(f"Error merging landmarks: {e}")
        return

    try:
        save_jsonl_gz(merged_landmarks_path, merged_data)
        print(f"Merged landmarks saved to {merged_landmarks_path}.")
    except Exception as e:
        print(f"Error saving merged landmarks: {e}")
