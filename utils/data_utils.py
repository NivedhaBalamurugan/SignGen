import os
import json
import numpy as np
from config import *

def load_skeleton_sequences(filepaths):
    skeleton_data = {}
    
    for filepath in filepaths:
        if not os.path.exists(filepath):
            logging.error(f"Skeleton data file not found: {filepath}")
            continue
            
        try:
            with open(filepath, 'r') as file:
                for line in file:
                    data = json.loads(line.strip())
                    word = list(data.keys())[0]
                    videos = data[word]
                    
                    if word not in skeleton_data:
                        skeleton_data[word] = []
                    
                    skeleton_data[word].extend([pad_video(video, MAX_FRAMES) for video in videos])
            logging.info(f"Processed {filepath}")
        except Exception as e:
            logging.error(f"Error processing {filepath}: {e}")

    return skeleton_data

def pad_video(video, max_frames):
    padded_video = np.zeros((max_frames, NUM_JOINTS, NUM_COORDINATES))
    video = np.array(video)[:max_frames] 
    padded_video[:video.shape[0], :, :] = video  
    return padded_video

def prepare_training_data(skeleton_data, word_embeddings):
    words = [word for word in skeleton_data.keys() if word in word_embeddings]
    if not words:
        logging.error("No matching words found between embeddings and skeleton data")
        return None, None

    all_skeleton_sequences = []
    all_word_vectors = []
    for word in words:
        for video in skeleton_data[word]:
            all_skeleton_sequences.append(video)
            all_word_vectors.append(word_embeddings[word])

    return np.array(all_skeleton_sequences), np.array(all_word_vectors)