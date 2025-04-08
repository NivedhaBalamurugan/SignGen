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
            merged_landmarks = np.vstack((palm_landmarks, body_landmarks))
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
    for item in processed_data:
        if item["gloss"] == word:
            video_path = item["video_path"]
            start_frame = item["frame_start"]
            end_frame = item["frame_end"]

            merged ,frame_count = get_video_landmarks(video_path, start_frame, end_frame)


def save_landmarks_to_jsonl(word_list, output_file):
    with gzip.open(output_file, 'wb') as f:
        for word in tqdm(word_list, desc="Processing words"):
            try:
                video_data = get_landmarks(word)
                word_entry = {word: video_data}
                f.write(orjson.dumps(word_entry) + b'\n')
            except Exception as e:
                print(f"Error processing {word}: {str(e)}")
                continue
            print(f"completed word {word}")

if __name__ == "__main__":
    word_list = [ "movie","police","boy","money","animal","flower","mirror","star","book", "now","table","black","doctor","hat","time","computer","fine",
        "chair","help","friend"]
    output_jsonl_path = "Dataset\new_words_extracted.jsonl.gz"  
    save_landmarks_to_jsonl(words_to_process, output_jsonl_path)
   