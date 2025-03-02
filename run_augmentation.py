import os
import json
import numpy as np
from collections import defaultdict
from augmentation import (generate_params, shear_landmarks, 
                        translate_landmarks, scale_landmarks)
from utils.jsonl_utils import load_jsonl_gz, save_jsonl_gz
from utils.data_utils import (denormalize_landmarks, normalize_landmarks)
from config import *

def save_chunk_statistics(stats, output_path):
    stats_path = output_path.replace('.jsonl.gz', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Chunk statistics saved to {stats_path}")

def augment_and_save(input_path, output_path):
    try:
        data = load_jsonl_gz(input_path)
        logging.info(f"Loaded {len(data)} words from {input_path}")
    except Exception as e:
        logging.error(f"Error loading JSONL file: {e}")
        return

    augmented_data = defaultdict(list)
    word_stats = {}
    total_videos = 0
    total_augmented = 0

    try:
        for word, videos in data.items():
            word_videos = len(videos)
            total_videos += word_videos
            logging.info(f"Processing word '{word}' with {word_videos} videos")
            
            word_stats[word] = {
                'original_videos': word_videos,
                'augmented_videos': word_videos * 6,
                'total_videos': word_videos * 7
            }

            for video_idx, video in enumerate(videos, 1):
                augmented_data[word].append(video)
                
                shearing_video, translation_video, scaling_video = [], [], []
                shearing_video_2, translation_video_2, scaling_video_2 = [], [], []
                
                shear_params, trans_params, scale_params = generate_params()
                shear_params_2, trans_params_2, scale_params_2 = generate_params()
                
                for frame in video:
                    frame = np.array(frame, dtype=np.float32)
                    frame = denormalize_landmarks(frame)
                    
                    sheared_frame = shear_landmarks(frame, shear_params)
                    translated_frame = translate_landmarks(frame, trans_params)
                    scaled_frame = scale_landmarks(frame, scale_params)
                    sheared_frame_2 = shear_landmarks(frame, shear_params_2)
                    translated_frame_2 = translate_landmarks(frame, trans_params_2)
                    scaled_frame_2 = scale_landmarks(frame, scale_params_2)
                    
                    sheared_frame = normalize_landmarks(sheared_frame)
                    translated_frame = normalize_landmarks(translated_frame)
                    scaled_frame = normalize_landmarks(scaled_frame)
                    sheared_frame_2 = normalize_landmarks(sheared_frame_2)
                    translated_frame_2 = normalize_landmarks(translated_frame_2)
                    scaled_frame_2 = normalize_landmarks(scaled_frame_2)
                    
                    shearing_video.append(sheared_frame.tolist())
                    translation_video.append(translated_frame.tolist())
                    scaling_video.append(scaled_frame.tolist())
                    shearing_video_2.append(sheared_frame_2.tolist())
                    translation_video_2.append(translated_frame_2.tolist())
                    scaling_video_2.append(scaled_frame_2.tolist())
                
                augmented_data[word].append(shearing_video)
                augmented_data[word].append(translation_video)
                augmented_data[word].append(scaling_video)
                augmented_data[word].append(shearing_video_2)
                augmented_data[word].append(translation_video_2)
                augmented_data[word].append(scaling_video_2)
                total_augmented += 6
                
                logging.info(f"Processed video {video_idx}/{word_videos} for word '{word}'")

        word_stats['__summary__'] = {
            'total_words': len(data),
            'total_original_videos': total_videos,
            'total_augmented_videos': total_augmented,
            'total_videos': total_videos + total_augmented
        }

        save_jsonl_gz(output_path, augmented_data, single_object=True)
        
        save_chunk_statistics(word_stats, output_path)
        
        logging.info(f"Successfully processed {total_videos} original videos")
        logging.info(f"Generated {total_augmented} augmented videos")
        logging.info(f"Total videos in output: {total_videos + total_augmented}")
        logging.info(f"Augmented landmarks saved to {output_path}")

    except Exception as e:
        logging.error(f"Error during augmentation: {e}")
        return

def main():
    os.makedirs(AUGMENTED_PATH, exist_ok=True)

    for i in range(10):
        paths = get_paths(i)
        merged_json_path = paths['merged_jsonl']
        augmented_json_path = paths['augmented_jsonl']
        input_path = merged_json_path
        output_path = augmented_json_path
        augment_and_save(input_path, output_path)

if __name__ == "__main__":
    main()