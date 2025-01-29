import cv2
import numpy as np
import mediapipe as mp
import json
from tqdm import tqdm
import DataProcessing


NUM_LANDMARKS = 21 + 21 + 6 + 1  
MAX_FRAME_COUNT = 0  


mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0  
    return frame

def normalize_landmarks(landmarks, frame_width, frame_height):
    landmarks[:, 0] /= frame_width  
    landmarks[:, 1] /= frame_height  
    return landmarks

def get_frame_landmarks(frame):
    
   
    
    palm_landmarks = np.zeros((42, 3), dtype=np.float64)  
    body_landmarks = np.zeros((7, 3), dtype=np.float64)   

    
    results_hands = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results_hands.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            if i == 0:  
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    palm_landmarks[idx] = [landmark.x, landmark.y, landmark.z]
            elif i == 1:  
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    palm_landmarks[21 + idx] = [landmark.x, landmark.y, landmark.z]


    results_pose = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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

    if start_frame <= 1:
        start_frame = 1
    elif start_frame > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        start_frame = 1
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame < 0:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_index = 1
    all_palm_landmarks = []
    all_body_landmarks = []

    while cap.isOpened() and frame_index <= end_frame:
        res, frame = cap.read()
        if not res:
            break
        if frame_index >= start_frame:
            frame.flags.writeable = False
            palm_landmarks, body_landmarks = get_frame_landmarks(frame)
            all_palm_landmarks.append(palm_landmarks.tolist())
            all_body_landmarks.append(body_landmarks.tolist())
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    return all_palm_landmarks, all_body_landmarks


palm_skeleton_data = {}
body_skeleton_data = {}

for data in tqdm(DataProcessing.processed_data, ncols=100):
    video_path = data["video_path"]
    start_frame = data["frame_start"]
    end_frame = data["frame_end"]
    gloss = data["gloss"]

    if gloss not in palm_skeleton_data:
        palm_skeleton_data[gloss] = []
        body_skeleton_data[gloss] = []

    try:
        palm_landmarks, body_landmarks = get_video_landmarks(video_path, start_frame, end_frame)
        palm_skeleton_data[gloss].append(palm_landmarks)
        body_skeleton_data[gloss].append(body_landmarks)
        MAX_FRAME_COUNT = max(MAX_FRAME_COUNT, end_frame - start_frame + 1)
    except Exception as e:
        print(f"\nError processing video {video_path}\n{e}")

# Save to JSON files
with open('/Users/subramaniansenthilkumar/Desktop/FYP/palm_skeleton_data.json', 'w') as palm_file:
    json.dump(palm_skeleton_data, palm_file)

with open('/Users/subramaniansenthilkumar/Desktop/FYP/body_skeleton_data.json', 'w') as body_file:
    json.dump(body_skeleton_data, body_file)

print(f"Processed landmarks for {len(palm_skeleton_data)} glosses.")