import os
import json
import numpy as np
from config import *

def load_glove_embeddings():    
        
    if not os.path.exists(GLOVE_TXT_PATH):
        logging.error(f"Word embeddings file not found: {GLOVE_TXT_PATH}")
        return None
        
    word_embeddings = {}
    try:
        with open(GLOVE_TXT_PATH, encoding="utf8") as file:
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


def load_word_embeddings():

    if (EMBEDDING_DIM == 50):
        word_embeddings = load_glove_embeddings()
        return word_embeddings    
       
    with open(ONE_HOT_TXT_PATH, 'r') as file:
        data = json.load(file)
    
    word_counts = {}
    for word, details in data.items():
        if word == "__summary__":
            continue
        original_videos = details.get('original_videos', 0)
        word_counts[word] = original_videos
    
    top_words = sorted(word_counts, key=word_counts.get, reverse=True)[:20]
    
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
                    
                    for video in videos:
                        video_sequence = np.zeros((30, 29, 2))
                        
                        for frame_idx, frame in enumerate(video[:30]):
                            upper = frame[:7] 
                            
                            left_wrist = frame[7]
                            left_thumb_cmc = frame[8]
                            left_thumb_tip = frame[11]
                            left_index_mcp = frame[12]
                            left_index_tip = frame[15]
                            left_middle_mcp = frame[16]
                            left_middle_tip = frame[19]
                            left_ring_mcp = frame[20]
                            left_ring_tip = frame[23]
                            left_pinky_mcp = frame[24]
                            left_pinky_tip = frame[27]
                            
                            right_wrist = frame[28]
                            right_thumb_mcp = frame[29]
                            right_thumb_tip = frame[32]
                            right_index_mcp = frame[33]
                            right_index_tip = frame[36]
                            right_middle_mcp = frame[37]
                            right_middle_tip = frame[40]
                            right_ring_mcp = frame[41]
                            right_ring_tip = frame[44]
                            right_pinky_mcp = frame[45]
                            right_pinky_tip = frame[48]
                            
                            frame_landmarks = np.concatenate([
                                upper, 
                                [left_wrist], [left_thumb_cmc], [left_thumb_tip], 
                                [left_index_mcp], [left_index_tip], [left_middle_mcp], [left_middle_tip], 
                                [left_ring_mcp], [left_ring_tip], [left_pinky_mcp], [left_pinky_tip],
                                [right_wrist], [right_thumb_mcp], [right_thumb_tip], 
                                [right_index_mcp], [right_index_tip], [right_middle_mcp], [right_middle_tip], 
                                [right_ring_mcp], [right_ring_tip], [right_pinky_mcp], [right_pinky_tip]
                            ])
                            
                            frame_landmarks_2d = frame_landmarks[:, :2]
                            
                            video_sequence[frame_idx] = frame_landmarks_2d
                        
                        skeleton_data[word].append(video_sequence)

        except Exception as e:
            logging.error(f"Error processing {filepath}: {str(e)}")
            continue

    for word in skeleton_data:
        skeleton_data[word] = np.array(skeleton_data[word])
        logging.info(f"Word '{word}' has shape {skeleton_data[word].shape}")
    
    return skeleton_data



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
    normalized = landmarks.copy()
    normalized[:, 0] = normalized[:, 0] / FRAME_WIDTH
    normalized[:, 1] = normalized[:, 1] / FRAME_HEIGHT
    
    return normalized

def denormalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    denormalized = landmarks.copy()
    denormalized[:, 0] = denormalized[:, 0] * FRAME_WIDTH
    denormalized[:, 1] = denormalized[:, 1] * FRAME_HEIGHT
    
    return denormalized


def select_sign_frames(original_frames):
    if len(original_frames) == 0:
        return []
    valid_frames = []
    
    for idx, frame in enumerate(original_frames):
        if not isinstance(frame, (list, np.ndarray)):
            continue
        try:
            if isinstance(frame, list):
                frame = np.array(frame, dtype=np.float32)
            if frame.ndim != 2 or frame.shape[0] != 49 or frame.shape[1] != 2:
                continue           

            upper = frame[:7]       # 0-6: Upper body
            hand1 = frame[7:28]     # 7-27: Hand 1
            hand2 = frame[28:49]    # 28-48: Hand 2
                
            hand1_valid = not np.all(hand1[0] == 0)
            hand2_valid = not np.all(hand2[0] == 0)
            
            hand1_zero = np.sum(np.all(hand1 == 0, axis=1)) 
            hand2_zero = np.sum(np.all(hand2 == 0, axis=1))
            
            hand1_mostly_present = hand1_valid and hand1_zero <= 10 
            hand2_mostly_present = hand2_valid and hand2_zero <= 10 
            
            if hand1_mostly_present or hand2_mostly_present:
                valid_frames.append(frame)
            # valid_frames.append(frame)
                
        except Exception as e:
            continue
    
    if len(valid_frames) < 20:
        return []
    elif len(valid_frames) == 30:
        return valid_frames
    elif len(valid_frames) > 30:
        indices = np.linspace(0, len(valid_frames) - 1, 30, dtype=int)
        selected_frames = [valid_frames[i] for i in indices]        
        return selected_frames
    else:
        num_frames_needed = 30 - len(valid_frames)
        repeat_indices = np.linspace(0, len(valid_frames) - 1, num_frames_needed, dtype=int)
        insert_operations = []

        for i in repeat_indices:
            insert_operations.append((i + 1, valid_frames[i]))
        insert_operations.sort(key=lambda x: x[0], reverse=True)
        
        result_frames = valid_frames.copy()
        
        for insert_idx, frame in insert_operations:
            if insert_idx <= len(result_frames):
                result_frames.insert(insert_idx, frame)
            else:
                result_frames.append(frame)        
        return result_frames