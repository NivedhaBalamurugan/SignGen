import json
import numpy as np  
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
        return data[key][0]  
    else:
        return None

def main(word):
    file_path = 'Dataset/landmarks/final/extracted_merged.jsonl' 
    data = read_jsonl_file(file_path)
    video_data = get_video(data, word)
    frames = np.array(video_data)
    return frames

if __name__ == "__main__":
    video_data = main("many")
    frames = np.array(video_data)
    print(frames.shape)
    show_output.save_generated_sequence(frames, CGAN_OUTPUT_FRAMES, CGAN_OUTPUT_VIDEO)      #without key frames    
    print(frames.shape)
    key_frames = np.array(select_sign_frames(frames))
    print(key_frames.shape)
    show_output.save_generated_sequence(key_frames, CVAE_OUTPUT_FRAMES, CVAE_OUTPUT_VIDEO)       #with key frames
    