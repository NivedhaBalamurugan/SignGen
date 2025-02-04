import cv2
import numpy as np
import mediapipe as mp
import json
import os
from tqdm import tqdm
import data_processing

PALM_JSON_PATH = "palm_landmarks.json"
BODY_JSON_PATH = "body_landmarks.json"
CHECKPOINT_PATH = "checkpoint.json"

NUM_LANDMARKS = 21 + 21 + 6 + 1

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    return frame

def normalize_landmarks(landmarks, frame_width, frame_height):
    landmarks[:, 0] /= frame_width  
    landmarks[:, 1] /= frame_height  
    return landmarks

def get_frame_landmarks(frame):
    palm_landmarks = np.zeros((42, 3), dtype=np.float64)
    body_landmarks = np.zeros((7, 3), dtype=np.float64)

    results_hands = hands.process(frame)
    if results_hands.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            if i < 2:
                offset = 21 * i
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    if idx < 21:
                        palm_landmarks[offset + idx] = [landmark.x, landmark.y, landmark.z]

    results_pose = pose.process(frame)
    if results_pose.pose_landmarks:
        upper_body_indices = [11, 12, 13, 14, 23, 24]
        for i, idx in enumerate(upper_body_indices):
            landmark = results_pose.pose_landmarks.landmark[idx]
            body_landmarks[i] = [landmark.x, landmark.y, landmark.z]
        
        face_landmark = results_pose.pose_landmarks.landmark[0]  
        body_landmarks[6] = [face_landmark.x, face_landmark.y, face_landmark.z]

    return palm_landmarks, body_landmarks

def get_video_landmarks(videoPath, start_frame, end_frame):
    cap = cv2.VideoCapture(videoPath)
    if start_frame < 1:
        start_frame = 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame < 0 or end_frame > total_frames:
        end_frame = total_frames

    all_palm_landmarks = []
    all_body_landmarks = []

    for frame_index in range(1, total_frames + 1):
        res, frame = cap.read()
        if not res:
            break
        if start_frame <= frame_index <= end_frame:
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = process_frame(frame)

            palm_landmarks, body_landmarks = get_frame_landmarks(frame)
            palm_landmarks = normalize_landmarks(palm_landmarks, frame.shape[1], frame.shape[0])
            body_landmarks = normalize_landmarks(body_landmarks, frame.shape[1], frame.shape[0])

            all_palm_landmarks.append(palm_landmarks.tolist())
            all_body_landmarks.append(body_landmarks.tolist())

    cap.release()
    return all_palm_landmarks, all_body_landmarks

def load_existing_data():
    if os.path.exists(PALM_JSON_PATH) and os.path.exists(BODY_JSON_PATH):
        with open(PALM_JSON_PATH, "r") as palm_file, open(BODY_JSON_PATH, "r") as body_file:
            try:
                palm_skeleton_data = json.load(palm_file)
                body_skeleton_data = json.load(body_file)
            except json.JSONDecodeError:
                palm_skeleton_data, body_skeleton_data = {}, {}
    else:
        palm_skeleton_data, body_skeleton_data = {}, {}

    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as checkpoint_file:
            try:
                processed_videos = set(json.load(checkpoint_file))
            except json.JSONDecodeError:
                processed_videos = set()
    else:
        processed_videos = set()

    return palm_skeleton_data, body_skeleton_data, processed_videos

def save_progress(palm_skeleton_data, body_skeleton_data, processed_videos):
    with open(PALM_JSON_PATH, "w") as palm_file:
        json.dump(palm_skeleton_data, palm_file, indent=4)
    with open(BODY_JSON_PATH, "w") as body_file:
        json.dump(body_skeleton_data, body_file, indent=4)
    with open(CHECKPOINT_PATH, "w") as checkpoint_file:
        json.dump(list(processed_videos), checkpoint_file, indent=4)

palm_skeleton_data, body_skeleton_data, processed_videos = load_existing_data()

for data in tqdm(data_processing.processed_data, ncols=100):
    video_path = data["video_path"]
    start_frame = data["frame_start"]
    end_frame = data["frame_end"]
    gloss = data["gloss"]

    if video_path in processed_videos:
        continue

    if gloss not in palm_skeleton_data:
        palm_skeleton_data[gloss] = []
        body_skeleton_data[gloss] = []

    try:
        palm_landmarks, body_landmarks = get_video_landmarks(video_path, start_frame, end_frame)
        palm_skeleton_data[gloss].append(palm_landmarks)
        body_skeleton_data[gloss].append(body_landmarks)

        processed_videos.add(video_path)
        save_progress(palm_skeleton_data, body_skeleton_data, processed_videos)

    except Exception as e:
        print(f"\nError processing video {video_path}: {e}")

print(f"Processed landmarks for {len(palm_skeleton_data)} glosses.")