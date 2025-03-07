import json
import numpy as np  # Ensure NumPy is imported
import show_output
from config import *
from utils.data_utils import select_sign_frames


def read_jsonl_file(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.update(item)
    return data

def get_video(data, key):
    if key in data and len(data[key]) >= 2:
        return data[key][0]  # Get the second video (index 1)
    else:
        return None

# Path to your JSONL file
file_path = 'Dataset/landmarks/final/0_landmarks_top20_split1_aug.jsonl' 

# Read the JSONL file
data = read_jsonl_file(file_path)

# Get the second video for the key "afternoon"
video_data = get_video(data, 'after')

if video_data is not None:
    aug_video = np.array(video_data)  # Convert only if video_data is not None
    print("Second video for 'afternoon':", aug_video.shape)

    # print(aug_video[1])
    key_frames = select_sign_frames(aug_video)
    show_output.save_generated_sequence(key_frames, CVAE_OUTPUT_FRAMES, CVAE_OUTPUT_VIDEO)
else:
    print("Key 'afternoon' not found or does not have at least two videos.")
