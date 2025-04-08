import cv2
import numpy as np
import mediapipe as mp
import orjson
import os
from tqdm import tqdm
from data_processing import processed_data
from collections import defaultdict
from utils.jsonl_utils import load_jsonl_gz, save_jsonl_gz
from config import *
import gzip

def extract_glove_embeddings(glove_file_path, output_file_path, target_words):
    target_words = set(target_words) if isinstance(target_words, list) else target_words
    found_words = set()
    
    with open(glove_file_path, 'r', encoding='utf-8') as glove_file, \
         open(output_file_path, 'w', encoding='utf-8') as out_file:
        
        for line in glove_file:
            parts = line.split()
            if not parts:
                continue
            word = parts[0]
            if word in target_words:
                out_file.write(line)
                found_words.add(word)
                if len(found_words) == len(target_words):
                    break
    
    # Check for missing words
    missing_words = target_words - found_words
    if missing_words:
        print(f"Warning: Missing embeddings for {len(missing_words)} words:")
        print(', '.join(sorted(missing_words)))
    else:
        print(f"Successfully extracted all {len(target_words)} word embeddings")




mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def process_frame(frame):
    frame = cv2.resize(frame, (FRAME_HEIGHT, FRAME_WIDTH))
    return frame


def normalize_landmarks(landmarks, frame_width, frame_height):
    landmarks[:, 0] /= frame_width
    landmarks[:, 1] /= frame_height
    return landmarks.astype(FP_PRECISION)


def get_frame_landmarks(frame):
    palm_landmarks = np.zeros((42, 2), dtype=FP_PRECISION)
    body_landmarks = np.zeros((7, 2), dtype=FP_PRECISION)

    results_hands = hands.process(frame)
    if results_hands.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            if i < 2:
                offset = 21 * i
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    if idx < 21:
                        palm_landmarks[offset + idx] = [landmark.x, landmark.y]

    results_pose = pose.process(frame)
    if results_pose.pose_landmarks:
        upper_body_indices = [11, 12, 13, 14, 23, 24]
        for i, idx in enumerate(upper_body_indices):
            landmark = results_pose.pose_landmarks.landmark[idx]
            body_landmarks[i] = [landmark.x, landmark.y]

        face_landmark = results_pose.pose_landmarks.landmark[0]
        body_landmarks[6] = [face_landmark.x, face_landmark.y]

    palm_landmarks = np.round(palm_landmarks, 5).astype(FP_PRECISION)
    body_landmarks = np.round(body_landmarks, 5).astype(FP_PRECISION)

    return palm_landmarks, body_landmarks


def get_video_landmarks(videoPath, start_frame, end_frame):
    cap = cv2.VideoCapture(videoPath)
    if start_frame < 1:
        start_frame = 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame < 0 or end_frame > total_frames:
        end_frame = total_frames

    frame_count = end_frame - start_frame + 1

    all_merged_landmarks = []  # To store merged landmarks for each frame

    # # Set window to be resizable
    # cv2.namedWindow('Video with Landmarks', cv2.WINDOW_NORMAL)

    for frame_index in range(1, total_frames + 1):
        res, frame = cap.read()
        if not res:
            break

        if start_frame <= frame_index <= end_frame:
            # Process the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = process_frame(frame_rgb)

            # Get landmarks
            frame.flags.writeable = False
            palm_landmarks, body_landmarks = get_frame_landmarks(processed_frame)
            frame.flags.writeable = True

            # Merge palm and body landmarks for the current frame
            merged_landmarks = np.vstack((body_landmarks, palm_landmarks))
            merged_landmarks = np.round(merged_landmarks, 5).astype(FP_PRECISION)
            all_merged_landmarks.append(merged_landmarks.tolist())


            # # Create a copy of the frame for drawing
            # display_frame = frame.copy()
            # frame_height, frame_width = display_frame.shape[:2]

            # # Draw merged landmarks
            # for landmark in merged_landmarks:
            #     if np.any(landmark):  # Check if the landmark is valid
            #         x, y = int(landmark[0] * frame_width), int(landmark[1] * frame_height)
            #         cv2.circle(display_frame, (x, y), 5, (0, 255, 255), -1)  # Yellow color for merged landmarks

            # # Add frame info
            # cv2.putText(display_frame, f"Frame: {frame_index}/{total_frames}", (20, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # # Display the frame with landmarks
            # cv2.imshow('Video with Landmarks', display_frame)

            # Control playback speed (30ms delay, adjust as needed)
            key = cv2.waitKey(30)
            if key & 0xFF == ord('q'):  # Press 'q' to quit early
                break
            elif key & 0xFF == ord('p'):  # Press 'p' to pause/play
                cv2.waitKey(0)  # Wait indefinitely until another key is pressed

    cap.release()
    cv2.destroyAllWindows()
    return all_merged_landmarks, frame_count

def get_landmarks(word):
    merged_all_videos = []
    for item in processed_data:
        if item["gloss"] == word:
            video_path = item["video_path"]
            start_frame = item["frame_start"]
            end_frame = item["frame_end"]
            merged ,frame_count = get_video_landmarks(video_path, start_frame, end_frame)
            merged_all_videos.append(merged)
    return merged_all_videos


def ensure_precision(data, decimals=5):
    if isinstance(data, (list, np.ndarray)):
        return [ensure_precision(x, decimals) for x in data]
    elif isinstance(data, (float, np.floating)):
        return float(f"{data:.{decimals}f}")
    elif isinstance(data, dict):
        return {k: ensure_precision(v, decimals) for k, v in data.items()}
    return data

def save_landmarks_to_jsonl(word_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, word in enumerate(tqdm(word_list, desc="Processing words"), start=1):
        output_file = os.path.join(output_dir, f"{idx}.jsonl.gz")
        try:
            video_data = get_landmarks(word)
            video_data = ensure_precision(video_data)
            word_entry = {word: video_data}
            with gzip.open(output_file, 'wb') as f:
                f.write(orjson.dumps(word_entry) + b'\n')
            print(f"Completed word {word} (saved to {output_file})")
        except Exception as e:
            print(f"Error processing {word}: {str(e)}")
            continue

if __name__ == "__main__":
    word_list = [ "movie","police","boy","money","animal","flower","mirror","star","book", "now","table","black","doctor","hat","time","computer","fine",
        "chair","help","friend"]
    output_dir = "Dataset/word_files"  
    save_landmarks_to_jsonl(word_list, output_dir)


    # glove_file = 'Dataset/glove/glove.6B.50d.txt'  
    # output_file = 'Dataset/custom_glove_20words.50d.txt'
    # extract_glove_embeddings(glove_file, output_file, word_list)
