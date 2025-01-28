import cv2
import numpy as np
import os
import DataProcessing
from tqdm import tqdm

NUM_LANDMARKS = 21+21+6+1 # 21 left hand(0-20), 21 right hand(0-20), 6 upperbody(11,12,13,14,23,24) + 1 face(0)
MAX_FRAME_COUNT = 0 #to finally pad the frames to this length

def process_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    return frame

def normalize_landmarks(landmarks, frame_width, frame_height):
    landmarks[:, 0] /= frame_width
    landmarks[:, 1] /= frame_height
    return landmarks

def get_video_landmarks(videoPath,start_frame,end_frame):
    
    cap = cv2.VideoCapture(videoPath)

    if start_frame <= 1:
        start_frame = 1
    elif start_frame > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        start_frame = 1
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    if end_frame < 0:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_index = 1
    all_frame_landmarks = np.zeros((end_frame-start_frame+1, NUM_LANDMARKS, 3), dtype=np.float64)

    while cap.isOpened() and frame_index <= end_frame:
        
        res, frame = cap.read()
        if res == False:
            break
        if frame_index >= start_frame:
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = process_frame(frame)
            # frame_landmarks = get_frame_landmarks(frame)  #complete this function which will return the landmarks for this frame using mediapipe
            # frame_landmarks = normalize_landmarks(frame_landmarks, frame.shape[1], frame.shape[0])
            # all_frame_landmarks[frame_index-start_frame] = frame_landmarks
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    return all_frame_landmarks



word_landmarks_map = {}

for data in tqdm(DataProcessing.processed_data, ncols=100):
    video_path = data["video_path"]
    start_frame = data["frame_start"]
    end_frame = data["frame_end"]
    gloss = data["gloss"]
    
    if gloss not in word_landmarks_map:
        word_landmarks_map[gloss] = []

    try:
        video_landmarks = get_video_landmarks(video_path, start_frame, end_frame)
        word_landmarks_map[gloss].append(video_landmarks)
        MAX_FRAME_COUNT = max(MAX_FRAME_COUNT, end_frame-start_frame+1)
    except Exception as e:
        print(f"\nError Encoding video {video_path}\n{e}")    


print(len(word_landmarks_map))