import os
import json
import numpy as np
from config import *
from utils.jsonl_utils import load_jsonl_gz

def load_glove_embeddings():    
        
    if not os.path.exists(GLOVE_TXT_PATH):
        logging.error(f"Word embeddings file not found: {GLOVE_TXT_PATH}")
        return None
        
    word_embeddings = {}
    try:
        with open(GLOVE_TXT_PATH, encoding="utf8") as file:
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespace
                if not line:  # Skip empty lines
                    continue
                    
                values = line.split()
                if len(values) <= 1:  # Skip lines without embeddings
                    continue
                    
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


import numpy as np

def select_sign_frames(original_frames, target_frames=30):
    def is_frame_valid(frame):
        try:
            if isinstance(frame, list):
                frame = np.array(frame, dtype=np.float32)
            
            if frame.ndim != 2 or frame.shape[0] != 49 or frame.shape[1] != 2:
                return False
            
            hand1 = frame[7:28]
            hand2 = frame[28:49]
            
            hand1_zero_count = np.sum(np.all(hand1 == 0, axis=1))
            hand2_zero_count = np.sum(np.all(hand2 == 0, axis=1))
            
            return (hand1_zero_count <= 10) or (hand2_zero_count <= 10)
        
        except Exception:
            return False

    valid_frames = [frame for frame in original_frames if is_frame_valid(frame)]
    
    if len(valid_frames) < target_frames // 2:
        return []
    
    if len(valid_frames) > target_frames:
        indices = np.linspace(0, len(valid_frames) - 1, target_frames, dtype=int)
        return [valid_frames[i] for i in indices]
    
    interpolated_frames = []
    indices = np.linspace(0, len(valid_frames) - 1, target_frames, dtype=float)
    
    for idx in indices:
        lower_idx = int(np.floor(idx))
        upper_idx = min(int(np.ceil(idx)), len(valid_frames) - 1)
        
        if lower_idx == upper_idx:
            interpolated_frames.append(valid_frames[lower_idx])
        else:
            lower_frame = np.array(valid_frames[lower_idx])
            upper_frame = np.array(valid_frames[upper_idx])
            weight = idx - lower_idx
            interpolated_frame = lower_frame * (1 - weight) + upper_frame * weight
            interpolated_frames.append(interpolated_frame.tolist())
    
    return interpolated_frames
    
def load_skeleton_sequences(filepaths):
    skeleton_data = {}

    for filepath in filepaths:
        if not os.path.exists(filepath):
            logging.error(f"Skeleton data file not found: {filepath}")
            continue

        try:
            data = load_jsonl_gz(filepath, single_object=False)
            
            for word, videos in data.items():
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

joint_connections = [
    # Upper body connections
    (0,1),   # Right shoulder to left shoulder
    (1,3),   # Left shoulder to left elbow
    (0,2),   # Right shoulder to right elbow 
    (1,5),   # Left shoulder to left hip
    (0,4),   # Right shoulder to right hip
    
    (3,7),   # Left elbow to left wrist
    (2,18),  # Right elbow to right wrist

    # Left hand connections
    (7, 8),   # Left wrist to left thumb CMC (base)
    (8, 9),   # Left thumb CMC to left thumb tip
    (7, 10),  # Left wrist to left index MCP (base)
    (10, 11), # Left index MCP to left index tip
    (7, 12),  # Left wrist to left middle MCP (base)
    (12, 13), # Left middle MCP to left middle tip
    (7, 14),  # Left wrist to left ring MCP (base)
    (14, 15), # Left ring MCP to left ring tip
    (7, 16),  # Left wrist to left pinky MCP (base)
    (16, 17), # Left pinky MCP to left pinky tip
    
    # Right hand connections
    (18, 19), # Right wrist to right thumb MCP (base)
    (19, 20), # Right thumb MCP to right thumb tip
    (18, 21), # Right wrist to right index MCP (base)
    (21, 22), # Right index MCP to right index tip
    (18, 23), # Right wrist to right middle MCP (base)
    (23, 24), # Right middle MCP to right middle tip
    (18, 25), # Right wrist to right ring MCP (base)
    (25, 26), # Right ring MCP to right ring tip
    (18, 27), # Right wrist to right pinky MCP (base)
    (27, 28)  # Right pinky MCP to right pinky tip
]

        
joint_angle_limits = [
    # Elbow angles
    (0, 2, 18, 0, 160),  # Right elbow angle (right shoulder - right elbow - right wrist)
    (1, 3, 7, 0, 160),   # Left elbow angle (left shoulder - left elbow - left wrist)
    
    # Shoulder angles
    (4, 0, 2, 0, 180),   # Right shoulder to elbow (right hip - right shoulder - right elbow)
    (5, 1, 3, 0, 180),   # Left shoulder to elbow (left hip - left shoulder - left elbow)
    (1, 0, 2, 0, 180),   # Across shoulders to right elbow (left shoulder - right shoulder - right elbow)
    (0, 1, 3, 0, 180),   # Across shoulders to left elbow (right shoulder - left shoulder - left elbow)
    
    # Wrist angles
    (2, 18, 19, 0, 90),  # Right wrist to thumb (right elbow - right wrist - right thumb base)
    (2, 18, 21, 0, 90),  # Right wrist to index (right elbow - right wrist - right index base)
    (3, 7, 8, 0, 90),    # Left wrist to thumb (left elbow - left wrist - left thumb base)
    (3, 7, 10, 0, 90),   # Left wrist to index (left elbow - left wrist - left index base)
    
    # Left hand finger angles
    (7, 8, 9, 0, 90),    # Left thumb angles (wrist - thumb base - thumb tip)
    (7, 10, 11, 0, 90),  # Left index finger (wrist - index base - index tip)
    (7, 12, 13, 0, 90),  # Left middle finger (wrist - middle base - middle tip)
    (7, 14, 15, 0, 90),  # Left ring finger (wrist - ring base - ring tip)
    (7, 16, 17, 0, 90),  # Left pinky finger (wrist - pinky base - pinky tip)
    
    # Right hand finger angles
    (18, 19, 20, 0, 90), # Right thumb angles (wrist - thumb base - thumb tip)
    (18, 21, 22, 0, 90), # Right index finger (wrist - index base - index tip)
    (18, 23, 24, 0, 90), # Right middle finger (wrist - middle base - middle tip)
    (18, 25, 26, 0, 90), # Right ring finger (wrist - ring base - ring tip)
    (18, 27, 28, 0, 90)  # Right pinky finger (wrist - pinky base - pinky tip)
]

def convert_skeleton_to_line_segments(skeleton_data):
    num_segments = len(joint_connections)
    line_segment_data = {}
    
    for word, videos in skeleton_data.items():
        num_videos = videos.shape[0]
        
        line_segments = np.zeros((num_videos, 30, num_segments, 2, 2))
        
        for video_idx in range(num_videos):
            for frame_idx in range(30):
                for segment_idx, (joint1_idx, joint2_idx) in enumerate(joint_connections):
                    
                    joint1 = videos[video_idx, frame_idx, joint1_idx]
                    joint2 = videos[video_idx, frame_idx, joint2_idx]
                    
                    line_segments[video_idx, frame_idx, segment_idx, 0] = joint1
                    line_segments[video_idx, frame_idx, segment_idx, 1] = joint2
        
        line_segment_data[word] = line_segments
    
    return line_segment_data, num_segments


def convert_line_segments_to_skeleton(line_segment_data):
    skeleton_data = {}
    
    for word, videos in line_segment_data.items():
        num_videos = videos.shape[0]
        num_frames = videos.shape[1]
        skeletons = np.zeros((num_videos, num_frames, 29, 2))
        
        for video_idx in range(num_videos):
            for frame_idx in range(num_frames):
                joint_positions = {}
                
                for segment_idx, (joint1_idx, joint2_idx) in enumerate(joint_connections):
                    joint1_pos = videos[video_idx, frame_idx, segment_idx, 0]
                    joint2_pos = videos[video_idx, frame_idx, segment_idx, 1]
                    
                    if joint1_idx not in joint_positions:
                        joint_positions[joint1_idx] = joint1_pos
                    if joint2_idx not in joint_positions:
                        joint_positions[joint2_idx] = joint2_pos
                
                for joint_idx in range(29):
                    if joint_idx in joint_positions:
                        skeletons[video_idx, frame_idx, joint_idx] = joint_positions[joint_idx]
        
        skeleton_data[word] = skeletons
    
    return skeleton_data

def convert_line_segments_to_skeleton_cvae(line_segment_data):
    num_frames, num_segments, num_joints, num_coords = line_segment_data.shape
    num_joints_total = 29  # Total joints in skeleton
    skeletons = np.zeros((num_frames, num_joints_total, num_coords))  

    for frame_idx in range(num_frames):
        joint_positions = {}

        for segment_idx, (joint1_idx, joint2_idx) in enumerate(joint_connections):
            joint1_pos = line_segment_data[frame_idx, segment_idx, 0, :]  # (x, y) of joint1
            joint2_pos = line_segment_data[frame_idx, segment_idx, 1, :]  # (x, y) of joint2

            joint_positions[joint1_idx] = joint1_pos
            joint_positions[joint2_idx] = joint2_pos

        for joint_idx in range(num_joints_total):
            if joint_idx in joint_positions:
                skeletons[frame_idx, joint_idx] = joint_positions[joint_idx]

    return skeletons