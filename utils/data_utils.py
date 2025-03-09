import os
import json
import numpy as np
from config import *
from scipy.spatial.distance import cdist
from ex import main
import show_output

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

def add_noise(frame, noise_level=0.001, consistent_noise=None):
    data_range = np.max(frame) - np.min(frame)
    scaled_noise_level = noise_level * data_range
    if consistent_noise is not None:
        noise = consistent_noise
    else:
        noise = np.random.normal(0, scaled_noise_level, frame.shape)
    noisy_frame = frame + noise
    noisy_frame = np.clip(noisy_frame, 0, 1)
    return noisy_frame, noise

def distort_frame(frame, noise_level, body_noise_level, body_noise=None):
    upper = frame[:7]       
    hand1 = frame[7:28]     
    hand2 = frame[28:49]    
    hand1_present = not np.all(hand1 == 0)
    hand2_present = not np.all(hand2 == 0)
    upper_noisy, body_noise = add_noise(upper, noise_level, consistent_noise=body_noise)
    hand1_noisy, _ = add_noise(hand1, noise_level) if hand1_present else (hand1, None)
    hand2_noisy, _ = add_noise(hand2, noise_level) if hand2_present else (hand2, None)
    noisy_frame = np.vstack((upper_noisy, hand1_noisy, hand2_noisy))
    return noisy_frame, body_noise


def get_cvae_sequences(word, isSave_Video, key_frames):
    model_path = os.path.join(CVAE_MODEL_PATH, "cvae.pth")
    if not os.path.exists(model_path):
        logging.error("Trained model file not found.")
        return None
    cvae_frames = []
    body_noise = None  
    noise_level=0.08
    body_noise_level=0.001
    for frame in key_frames:
        distorted_frame, body_noise = distort_frame(frame, noise_level, body_noise_level, body_noise)
        cvae_frames.append(distorted_frame)
    if isSave_Video:
        show_output.save_generated_sequence(cvae_frames, CVAE_OUTPUT_FRAMES, CVAE_OUTPUT_VIDEO) 
    return cvae_frames

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
    valid_indices = []
    for idx, frame in enumerate(original_frames):
        if not isinstance(frame, (list, np.ndarray)):
            continue
        try:
            if isinstance(frame, list):
                frame = np.array(frame, dtype=np.float32)
            if frame.ndim != 2 or frame.shape[0] != 49 or frame.shape[1] != 2:
                continue
            if np.any(np.all(frame == 0, axis=1)):
                continue
            upper = frame[:7]       # 0-6: Upper body
            hand1 = frame[7:28]     # 7-27: Hand 1
            hand2 = frame[28:49]    # 28-48: Hand 2
            if np.any(np.all(upper == 0, axis=1)):
                continue
            if np.all(hand1[0] == 0) or np.all(hand2[0] == 0):
                continue
            hand1_zero = np.sum(np.all(hand1 == 0, axis=1)) 
            hand2_zero = np.sum(np.all(hand2 == 0, axis=1))
            if (hand1_zero > 10 or hand2_zero > 10 or 
                np.all(hand1 == 0) or np.all(hand2 == 0)):
                continue
            valid_frames.append(frame)
            valid_indices.append(idx)
        except Exception as e:
            continue
    if len(valid_frames) == 0:
        return []
    motion_scores = np.zeros(len(valid_frames))
    hand_joints = list(range(7, 49))  
    prev_hands = None
    for i, frame in enumerate(valid_frames):
        current_hands = frame[hand_joints]
        if prev_hands is not None:
            dists = np.linalg.norm(current_hands - prev_hands, axis=1)
            motion_scores[i] = np.sum(dists)
        prev_hands = current_hands
    if len(motion_scores) <= 30:
        selected_indices = np.arange(len(motion_scores))
    else:
        selected_indices = np.argsort(motion_scores)[-30:]
    selected_indices = np.sort(selected_indices)  
    selected_frames = [valid_frames[i] for i in selected_indices]
    if len(selected_frames) < 30:
        num_frames_needed = 30 - len(selected_frames)
        if len(selected_frames) > 0: 
            repeat_indices = np.linspace(0, len(selected_frames) - 1, num_frames_needed, dtype=int)
            repeated_frames = [selected_frames[i] for i in repeat_indices]
            selected_frames.extend(repeated_frames)
            selected_frames = [selected_frames[i] for i in np.argsort(np.concatenate([selected_indices, repeat_indices]))]
        else:
            return []   
    return selected_frames[:30]  