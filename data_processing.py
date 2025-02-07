import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


with open('Dataset/WLASL_v0.3.json') as json_file:
    all_data = json.load(json_file)

with open('Dataset/missing.txt') as missing_file:
    missing_video_ids = missing_file.read().splitlines()

video_dir = "Dataset/videos"

processed_data = []

MAX_FRAME_COUNT = 0  
total_videos = 0
missing_videos = 0
video_lengths = []
gloss_frequency = {}


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
            MAX_FRAME_COUNT = max(MAX_FRAME_COUNT, frame_count)
            video_lengths.append(frame_count)
            
            if gloss in gloss_frequency:
                gloss_frequency[gloss] += 1
            else:
                gloss_frequency[gloss] = 1

# print(f"Total videos: {total_videos}")
# print(f"Missing videos: {missing_videos}")
# print(f"Processed videos: {len(processed_data)}")
# print(f"Max frame count: {MAX_FRAME_COUNT}")

# if processed_data:
#     print("\nStructure of processed_data:")
#     print(json.dumps(processed_data[0], indent=4))
# else:
#     print("No data processed.")

# plt.figure(figsize=(10, 6))
# sns.histplot(video_lengths, bins=50, kde=True)
# plt.title('Distribution of Video Lengths (in Frames)')
# plt.xlabel('Frame Count')
# plt.ylabel('Frequency')
# plt.show()

# N = 20  # Number of top glosses to display
# sorted_gloss_frequency = sorted(gloss_frequency.items(), key=lambda x: x[1], reverse=True)[:N]
# glosses, frequencies = zip(*sorted_gloss_frequency)

# plt.figure(figsize=(12, 8))
# sns.barplot(x=list(frequencies), y=list(glosses), palette="viridis")
# plt.title(f'Top {N} Most Frequent Glosses')
# plt.xlabel('Frequency')
# plt.ylabel('Gloss')
# plt.show()

# labels = ['Processed Videos', 'Missing Videos']
# sizes = [len(processed_data), missing_videos]
# colors = ['#66b3ff', '#ff9999']

# plt.figure(figsize=(6, 6))
# plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
# plt.title('Processed vs Missing Videos')
# plt.show()