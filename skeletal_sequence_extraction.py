import cv2
import numpy as np
import mediapipe as mp
import orjson
import os
from tqdm import tqdm
import data_processing
from collections import defaultdict
from utils.jsonl_utils import load_jsonl_gz, save_jsonl_gz
from config import *

CHUNK_INDEX = 0
paths = get_paths(CHUNK_INDEX)
PALM_JSONL_PATH = paths['palm_jsonl']
BODY_JSONL_PATH = paths['body_jsonl']
CHECKPOINT_PATH = paths['checkpoint']

MAX_FRAME_COUNT = 0
SAVE_EVERY_N_VIDEOS = 10
NUM_LANDMARKS = 21 + 21 + 6 + 1

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

            all_palm_landmarks.append(palm_landmarks.tolist())
            all_body_landmarks.append(body_landmarks.tolist())

    cap.release()
    return all_palm_landmarks, all_body_landmarks, frame_count


def load_jsonl_gz_as_dict(file_path):
    return load_jsonl_gz(file_path, single_object=True)


def save_grouped_jsonl_gz(file_path, data):
    save_jsonl_gz(file_path, data, single_object=True)


def load_existing_data():
    palm_skeleton_data = load_jsonl_gz_as_dict(PALM_JSONL_PATH)
    body_skeleton_data = load_jsonl_gz_as_dict(BODY_JSONL_PATH)

    processed_videos = set()
    max_frame_count = 0

    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as checkpoint_file:
            try:
                checkpoint_data = orjson.loads(checkpoint_file.read())
                processed_videos = set(checkpoint_data.get("processed_videos", []))
                max_frame_count = checkpoint_data.get("max_frame_count", 0)
            except orjson.JSONDecodeError:
                pass

    return palm_skeleton_data, body_skeleton_data, processed_videos, max_frame_count


def save_checkpoint(processed_videos, max_frame_count):
    checkpoint_data = {
        "processed_videos": list(processed_videos),
        "max_frame_count": max_frame_count
    }
    with open(CHECKPOINT_PATH, "wb") as checkpoint_file:
        checkpoint_file.write(orjson.dumps(checkpoint_data))


def padding(total_skeleton_data, MAX_FRAME_COUNT):
    padded_total_skeleton_data = defaultdict(list)

    for gloss, videos in total_skeleton_data.items():
        for video in videos:
            padding = [[0,0,0]] * (MAX_FRAME_COUNT - len(video))
            padded_video = video + padding
            padded_total_skeleton_data[gloss].append(padded_video)
    return padded_total_skeleton_data


palm_skeleton_data, body_skeleton_data, processed_videos, MAX_FRAME_COUNT = load_existing_data()
processed_count = 0

for data in tqdm(data_processing.processed_data, ncols=100):
    video_path = data["video_path"]
    start_frame = data["frame_start"]
    end_frame = data["frame_end"]
    gloss = data["gloss"]

    if video_path in processed_videos:
        continue

    try:
        palm_landmarks, body_landmarks, frame_count = get_video_landmarks(video_path, start_frame, end_frame)
        palm_landmarks = np.around(palm_landmarks, decimals=5).tolist()
        body_landmarks = np.around(body_landmarks, decimals=5).tolist()

        palm_skeleton_data[gloss].append(palm_landmarks)
        body_skeleton_data[gloss].append(body_landmarks)

        MAX_FRAME_COUNT = max(MAX_FRAME_COUNT, frame_count)

        processed_videos.add(video_path)
        processed_count += 1

        if processed_count % SAVE_EVERY_N_VIDEOS == 0:
            save_grouped_jsonl_gz(PALM_JSONL_PATH, palm_skeleton_data)
            save_grouped_jsonl_gz(BODY_JSONL_PATH, body_skeleton_data)
            save_checkpoint(processed_videos, MAX_FRAME_COUNT)
            logging.info(f"Saved progress after {processed_count} videos.")

    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")


os.makedirs(PALM_BODY_PATH, exist_ok=True)
save_grouped_jsonl_gz(PALM_JSONL_PATH, palm_skeleton_data)
save_grouped_jsonl_gz(BODY_JSONL_PATH, body_skeleton_data)
save_checkpoint(processed_videos, MAX_FRAME_COUNT)

logging.info(f"Maximum frame count encountered: {MAX_FRAME_COUNT}")