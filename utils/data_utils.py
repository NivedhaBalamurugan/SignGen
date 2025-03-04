import os
import json
import numpy as np
from config import *

def load_word_embeddings(filepath):    
        
    if not os.path.exists(filepath):
        logging.error(f"Word embeddings file not found: {filepath}")
        return None
        
    word_embeddings = {}
    try:
        with open(filepath, encoding="utf8") as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype=FP_PRECISION)
                word_embeddings[word] = vector
        logging.info(f"Loaded {len(word_embeddings)} word embeddings")
        return word_embeddings
    except Exception as e:
        logging.error(f"Error loading word embeddings: {e}")
        return None

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

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks to MediaPipe's format (0 to 1).
    Only normalize x and y coordinates, preserve z as it's already normalized.
    
    Args:
        landmarks: numpy array of shape (N, 3) where N is number of landmarks
    Returns:
        normalized landmarks of the same shape
    """
    normalized = landmarks.copy()
    normalized[:, 0] = normalized[:, 0] / FRAME_WIDTH
    normalized[:, 1] = normalized[:, 1] / FRAME_HEIGHT
    
    return normalized

def denormalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Convert normalized landmarks back to pixel coordinates.
    Only denormalize x and y coordinates, preserve z as it's relative depth.
    
    Args:
        landmarks: numpy array of shape (N, 3) where N is number of landmarks
    Returns:
        denormalized landmarks of the same shape
    """
    denormalized = landmarks.copy()
    denormalized[:, 0] = denormalized[:, 0] * FRAME_WIDTH
    denormalized[:, 1] = denormalized[:, 1] * FRAME_HEIGHT
    
    return denormalized