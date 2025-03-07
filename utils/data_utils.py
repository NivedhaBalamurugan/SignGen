import os
import json
import numpy as np
from config import *
from scipy.spatial.distance import cdist


def load_word_embeddings(filepath):    
       
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    # Count the number of original videos for each word
    word_counts = {}
    for word, details in data.items():
        if word == "__summary__":
            continue
        original_videos = details.get('original_videos', 0)
        word_counts[word] = original_videos
    
    # Select the top 20 words with the highest original video counts
    top_words = sorted(word_counts, key=word_counts.get, reverse=True)[:20]
    
    # Create one-hot encoding for the top words
    word_embeddings = {}
    for i, word in enumerate(top_words):
        one_hot_vector = np.zeros(len(top_words), dtype=np.float32)
        one_hot_vector[i] = 1.0
        word_embeddings[word] = one_hot_vector
    
    return word_embeddings



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



def select_sign_frames(original_frames):
    """
    Select the most informative frames from a sign video.
    Handles various input formats and potential structural issues.
    """
    # Guard against empty input
    if not original_frames or len(original_frames) == 0:
        return []
    
    # Ensure frames are in correct format
    valid_frames = []
    valid_indices = []
    
    for idx, frame in enumerate(original_frames):
        # Skip if frame is not a proper data structure
        if not isinstance(frame, (list, np.ndarray)):
            continue
            
        # Convert to numpy array for consistent handling
        try:
            if isinstance(frame, list):
                frame = np.array(frame, dtype=np.float32)
                
            # Skip frames with wrong dimensions
            if frame.ndim == 0 or frame.shape[0] != 49:
                continue
                
            # Check if any joint has all 3 coordinates as 0
            if np.any(np.all(frame == 0, axis=1)):
                continue
                
            # Split joints - make sure frame has correct structure first
            if frame.ndim < 2 or frame.shape[1] < 3:
                continue
                
            upper = frame[:7]       # 0-6: Upper body
            hand1 = frame[7:28]     # 7-27: Hand 1
            hand2 = frame[28:49]    # 28-48: Hand 2
            
            # Check upper body (all joints must be present)
            if np.any(np.all(upper == 0, axis=1)):
                continue
                
            # Check palm index (index 0 in hand 1 or hand 2)
            if np.all(hand1[0] == 0) or np.all(hand2[0] == 0):
                continue
                
            # Check hand validity
            hand1_zero = np.sum(np.all(hand1 == 0, axis=1)) 
            hand2_zero = np.sum(np.all(hand2 == 0, axis=1))
            
            if (hand1_zero > 10 or hand2_zero > 10 or 
                np.all(hand1 == 0) or np.all(hand2 == 0)):
                continue
                
            valid_frames.append(frame)
            valid_indices.append(idx)
        except Exception as e:
            # Skip frames that cause errors
            continue

    # Early exit if no valid frames
    if len(valid_frames) == 0:
        return []

    # Motion scoring (focus on hand trajectories)
    motion_scores = np.zeros(len(valid_frames))
    hand_joints = list(range(7, 49))  # All hand joints
    
    prev_hands = None
    for i, frame in enumerate(valid_frames):
        current_hands = frame[hand_joints]
        
        if prev_hands is not None:
            # Calculate per-joint movement magnitude
            dists = np.linalg.norm(current_hands - prev_hands, axis=1)
            motion_scores[i] = np.sum(dists)
            
        prev_hands = current_hands

    # Select frames with highest motion scores (up to 30)
    if len(motion_scores) <= 30:
        selected_indices = np.arange(len(motion_scores))
    else:
        selected_indices = np.argsort(motion_scores)[-30:]
    
    selected_indices = np.sort(selected_indices)  # Maintain temporal order
    selected_frames = [valid_frames[i] for i in selected_indices]

    # If fewer than 30 frames, uniformly repeat frames to make it 30
    if len(selected_frames) < 30:
        num_frames_needed = 30 - len(selected_frames)
        if len(selected_frames) > 0:  # Check to avoid division by zero
            repeat_indices = np.linspace(0, len(selected_frames) - 1, num_frames_needed, dtype=int)
            repeated_frames = [selected_frames[i] for i in repeat_indices]
            selected_frames.extend(repeated_frames)
            
            # Sort the final list to preserve temporal order
            selected_frames = [selected_frames[i] for i in np.argsort(np.concatenate([selected_indices, repeat_indices]))]
        else:
            return []

    return selected_frames[:30]  # Ensure exactly 30 frames