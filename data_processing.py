import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


with open('Dataset/WLASL_v0.3.json') as json_file:
    all_data = json.load(json_file)

with open('Dataset/missing.txt') as missing_file:
    missing_video_ids = missing_file.read().splitlines()

video_dir = "Dataset/videos"

processed_data = []
total_videos = 0
missing_videos = 0



for i in tqdm(range(len(all_data)), ncols=100):
    gloss = all_data[i]["gloss"]
    instances = all_data[i]["instances"]

    for instance in instances:
        video_id = instance["video_id"]
        total_videos += 1

        if video_id in missing_video_ids:
            missing_videos += 1
            continue
        
        if os.path.exists(os.path.join(video_dir, f'{video_id}.mp4')):
            video_path = os.path.join(video_dir, f'{video_id}.mp4')
            frame_start = instance["frame_start"]
            frame_end = instance["frame_end"]
            frame_count = frame_end - frame_start + 1
            processed_data.append({
                "gloss": gloss,
                "video_path": video_path,
                "frame_start": frame_start,
                "frame_end": frame_end
            })
            

print(f"Total videos: {total_videos}")
print(f"Missing videos: {missing_videos}")
print(f"Processed videos: {len(processed_data)}")

if processed_data:
    print("\nStructure of processed_data:")
    print(json.dumps(processed_data[0], indent=4))
else:
    print("No data processed.")

