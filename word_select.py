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

    return palm_landmarks, body_landmarks


def get_video_landmarks(videoPath, start_frame, end_frame):
    cap = cv2.VideoCapture(videoPath)
    if start_frame < 1:
        start_frame = 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame < 0 or end_frame > total_frames:
        end_frame = total_frames

    frame_count = end_frame - start_frame + 1
    
    # Set up colors for different landmarks
    HAND_COLORS = {
        'thumb': (0, 255, 0),       # Green
        'index': (255, 0, 0),       # Blue
        'middle': (0, 0, 255),      # Red
        'ring': (255, 255, 0),      # Cyan
        'pinky': (255, 0, 255)      # Magenta
    }
    
    BODY_COLORS = {
        'shoulders': (255, 165, 0),  # Orange
        'arms': (128, 0, 128),       # Purple
        'face': (255, 255, 255)      # White
    }

    all_palm_landmarks = []
    all_body_landmarks = []
    
    # Set window to be resizable
    cv2.namedWindow('Video with Landmarks', cv2.WINDOW_NORMAL)

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
            
            # Create a copy of the frame for drawing
            display_frame = frame.copy()
            frame_height, frame_width = display_frame.shape[:2]
            
            # Draw hand landmarks with different colors for each finger
            if np.any(palm_landmarks):
                # Define finger groups (thumb: 0-4, index: 5-8, etc.)
                finger_groups = {
                    'thumb': [0, 1, 2, 3, 4],
                    'index': [5, 6, 7, 8],
                    'middle': [9, 10, 11, 12],
                    'ring': [13, 14, 15, 16],
                    'pinky': [17, 18, 19, 20]
                }
                
                # For each hand (up to 2)
                for hand_idx in range(2):
                    offset = 21 * hand_idx
                    
                    # Draw each finger with its specific color
                    for finger, indices in finger_groups.items():
                        for idx in indices:
                            if np.any(palm_landmarks[offset + idx]):
                                x, y = int(palm_landmarks[offset + idx][0] * frame_width), int(palm_landmarks[offset + idx][1] * frame_height)
                                cv2.circle(display_frame, (x, y), 5, HAND_COLORS[finger], -1)
                                
                                # Connect joints with lines (within fingers)
                                if idx > 0 and idx % 4 != 0:  # Not the first joint of a finger
                                    prev_x, prev_y = int(palm_landmarks[offset + idx - 1][0] * frame_width), int(palm_landmarks[offset + idx - 1][1] * frame_height)
                                    cv2.line(display_frame, (x, y), (prev_x, prev_y), HAND_COLORS[finger], 2)
            
            # Draw body landmarks
            if np.any(body_landmarks):
                # Shoulders (indices 0-1)
                for i in range(2):
                    x, y = int(body_landmarks[i][0] * frame_width), int(body_landmarks[i][1] * frame_height)
                    cv2.circle(display_frame, (x, y), 7, BODY_COLORS['shoulders'], -1)
                
                # Draw line between shoulders
                if np.any(body_landmarks[0]) and np.any(body_landmarks[1]):
                    left_x, left_y = int(body_landmarks[0][0] * frame_width), int(body_landmarks[0][1] * frame_height)
                    right_x, right_y = int(body_landmarks[1][0] * frame_width), int(body_landmarks[1][1] * frame_height)
                    cv2.line(display_frame, (left_x, left_y), (right_x, right_y), BODY_COLORS['shoulders'], 2)
                
                # Arms (indices 2-5)
                for i in range(2, 6):
                    x, y = int(body_landmarks[i][0] * frame_width), int(body_landmarks[i][1] * frame_height)
                    cv2.circle(display_frame, (x, y), 7, BODY_COLORS['arms'], -1)
                    
                    # Connect joints (shoulders to elbows, elbows to wrists)
                    if i in [2, 4]:  # Left/right elbow
                        shoulder_idx = 0 if i == 2 else 1  # Left/right shoulder
                        shoulder_x, shoulder_y = int(body_landmarks[shoulder_idx][0] * frame_width), int(body_landmarks[shoulder_idx][1] * frame_height)
                        cv2.line(display_frame, (x, y), (shoulder_x, shoulder_y), BODY_COLORS['arms'], 2)
                    
                    if i in [3, 5]:  # Left/right wrist
                        elbow_idx = 2 if i == 3 else 4  # Left/right elbow
                        elbow_x, elbow_y = int(body_landmarks[elbow_idx][0] * frame_width), int(body_landmarks[elbow_idx][1] * frame_height)
                        cv2.line(display_frame, (x, y), (elbow_x, elbow_y), BODY_COLORS['arms'], 2)
                
                # Face (index 6)
                if np.any(body_landmarks[6]):
                    x, y = int(body_landmarks[6][0] * frame_width), int(body_landmarks[6][1] * frame_height)
                    cv2.circle(display_frame, (x, y), 10, BODY_COLORS['face'], -1)
                    
                    # Connect face to shoulders
                    if np.any(body_landmarks[0]) and np.any(body_landmarks[1]):
                        mid_shoulder_x = int((body_landmarks[0][0] + body_landmarks[1][0]) * frame_width / 2)
                        mid_shoulder_y = int((body_landmarks[0][1] + body_landmarks[1][1]) * frame_height / 2)
                        cv2.line(display_frame, (x, y), (mid_shoulder_x, mid_shoulder_y), BODY_COLORS['face'], 2)
            
            # Add frame info
            cv2.putText(display_frame, f"Frame: {frame_index}/{total_frames}", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Store landmarks
            all_palm_landmarks.append(palm_landmarks.tolist())
            all_body_landmarks.append(body_landmarks.tolist())

            # Display the frame with landmarks
            cv2.imshow('Video with Landmarks', display_frame)
            
            # Control playback speed (30ms delay, adjust as needed)
            key = cv2.waitKey(30) 
            if key & 0xFF == ord('q'):  # Press 'q' to quit early
                break
            elif key & 0xFF == ord('p'):  # Press 'p' to pause/play
                cv2.waitKey(0)  # Wait indefinitely until another key is pressed

    cap.release()
    cv2.destroyAllWindows()
    return all_palm_landmarks, all_body_landmarks, frame_count




word = "star"  # Replace with the word you want to search
for item in processed_data:
    if item["gloss"] == word:
        video_path = item["video_path"]
        start_frame = item["frame_start"]
        end_frame = item["frame_end"]

        get_video_landmarks(video_path, start_frame, end_frame)
