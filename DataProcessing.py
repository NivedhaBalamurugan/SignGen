import json
import os
from tqdm import tqdm

with open('Dataset/WLASL_v0.3.json') as json_file:
    all_data = json.load(json_file)

with open('Dataset/missing.txt') as missing_file:
    missing_video_ids = missing_file.read().splitlines()

video_dir = "Dataset/videos"

processed_data = []


for i in tqdm(range(len(all_data)), ncols=100):
    gloss = all_data[i]["gloss"]
    instances = all_data[i]["instances"]

    for instance in instances:
        video_id = instance["video_id"]

        if video_id in missing_video_ids:
            continue
        
        if os.path.exists(os.path.join(video_dir, f'{video_id}.mp4')):
            video_path = os.path.join(video_dir, f'{video_id}.mp4')
            frame_start = instance["frame_start"]
            frame_end = instance["frame_end"]
            processed_data.append({
                "gloss": gloss,
                "video_path": video_path,
                "frame_start": frame_start,
                "frame_end": frame_end
            })

print(len(processed_data))