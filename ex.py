import json
import numpy as np  
import show_output
from config import *
from utils.data_utils import *
from utils.jsonl_utils import *

def read_jsonl_file(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            for key in item:
                if key not in data:
                    data[key] = []
                data[key].extend(item[key])  # Changed from update to extend
    return data

def get_video(data, key):
    if not data or key not in data or not data[key]:
        print(f"Warning: No data found for word '{key}'")
        return None
    return data[key][0]

def main(word):
    file_path = 'Dataset/new_words_extracted.jsonl.gz' 
    data = load_jsonl_gz(file_path)
    video_data = get_video(data, word)
    frames = np.array(video_data)
    return frames

if __name__ == "__main__":
    video_data = main("movie")
    frames = np.array(video_data)
    print(frames.shape)
    key_frames = np.array(select_sign_frames(frames))
    # key_frames = np.array(select_sign_frames(frames))
    # print(key_frames.shape)
    # dict = {"fine" : np.array([frames])}
    # print("right shoulder ", frames[1][0])
    # print("left shoulder ", frames[1][1])
    # print("right hip ", frames[1][4])
    # print("left hip ", frames[1][5])
    # print("nose ", frames[1][6])
    show_output.save_generated_sequence(key_frames, CGAN_OUTPUT_FRAMES, CGAN_OUTPUT_VIDEO)      
    