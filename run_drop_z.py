import glob
import os
from utils.jsonl_utils import load_jsonl_gz, save_jsonl_gz
from config import logging

def drop_z_coordinate(data):
    processed_data = {}
    
    for gloss, videos in data.items():
        processed_videos = []
        for video in videos:
            processed_frames = []
            for frame in video:
                processed_landmarks = [landmark[:2] for landmark in frame]
                processed_frames.append(processed_landmarks)
            processed_videos.append(processed_frames)
        processed_data[gloss] = processed_videos
    
    return processed_data

def process_file(input_path, output_path):
    data = load_jsonl_gz(input_path)
    processed_data = drop_z_coordinate(data)
    save_jsonl_gz(output_path, processed_data, single_object=False)

def main():
    input_dir = f"Dataset\landmarks\split_augmentation_30"
    output_dir =  f"Dataset\landmarks\split_augmentation_30_no_z"
    
    os.makedirs(output_dir, exist_ok=True)
    jsonl_files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl.gz")))

    for file_path in jsonl_files:
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        logging.info(f"Processing {file_path}")
        process_file(str(file_path), str(output_path))
        logging.info(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
