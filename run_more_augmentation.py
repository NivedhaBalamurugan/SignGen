import os
import json
import numpy as np
from collections import defaultdict
from augmentation import (generate_params, shear_landmarks, 
                        translate_landmarks, scale_landmarks)
from utils.jsonl_utils import load_jsonl_gz, save_jsonl_gz
from utils.data_utils import (denormalize_landmarks, normalize_landmarks)
from config import *

def augment_to_target(input_path, output_path, target_videos=100):
    """Augment each word to have target number of videos."""
    try:
        data = load_jsonl_gz(input_path)
        logging.info(f"Loaded {len(data)} words from {input_path}")
    except Exception as e:
        logging.error(f"Error loading JSONL file: {e}")
        return

    augmented_data = defaultdict(list)
    stats = {}
    total_original = 0
    total_augmented = 0

    try:
        for word, videos in data.items():
            original_count = len(videos)
            total_original += original_count
            
            # Calculate how many copies we need
            augmentations_needed = max(0, target_videos - original_count)
            copies_per_video = -(-augmentations_needed // original_count)
            
            logging.info(f"Processing '{word}': {original_count} original videos, need {augmentations_needed} more")
            
            augmented_data[word].extend(videos)
            
            for video_idx, video in enumerate(videos, 1):
                for copy in range(copies_per_video):
                    if len(augmented_data[word]) >= target_videos:
                        break
                        
                    new_video = []
                    shear_params, trans_params, scale_params = generate_params()
                    
                    for frame in video:
                        frame = np.array(frame, dtype=np.float32)
                        frame = denormalize_landmarks(frame)
                        
                        # Apply random combination of augmentations
                        augmented_frame = frame
                        if np.random.random() > 0.3:
                            augmented_frame = shear_landmarks(augmented_frame, shear_params)
                        if np.random.random() > 0.3:
                            augmented_frame = translate_landmarks(augmented_frame, trans_params)
                        if np.random.random() > 0.3:
                            augmented_frame = scale_landmarks(augmented_frame, scale_params)
                            
                        augmented_frame = normalize_landmarks(augmented_frame)
                        new_video.append(augmented_frame.tolist())
                    
                    augmented_data[word].append(new_video)
                    total_augmented += 1
                    
                    if video_idx % 10 == 0:
                        logging.info(f"Created {len(augmented_data[word])}/{target_videos} videos for '{word}'")

            final_count = len(augmented_data[word])
            stats[word] = {
                'original_videos': original_count,
                'augmented_videos': final_count - original_count,
                'total_videos': final_count
            }
            
            logging.info(f"Completed '{word}': {original_count} -> {final_count} videos")

        # Save statistics
        stats['__summary__'] = {
            'total_words': len(data),
            'total_original_videos': total_original,
            'total_augmented_videos': total_augmented,
            'total_videos': total_original + total_augmented,
            'target_per_word': target_videos
        }
        
        stats_path = output_path.replace('.jsonl.gz', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Save augmented data
        save_jsonl_gz(output_path, augmented_data, single_object=True)
        
        logging.info(f"Successfully augmented dataset:")
        logging.info(f"Original videos: {total_original}")
        logging.info(f"Augmented videos: {total_augmented}")
        logging.info(f"Total videos: {total_original + total_augmented}")
        logging.info(f"Average per word: {(total_original + total_augmented) / len(data):.1f}")
        logging.info(f"Saved to: {output_path}")

    except Exception as e:
        logging.error(f"Error during augmentation: {e}")
        raise

def main():
    top_n = 20
    target_videos = 300

    setup_logging(f"augmentation_top{top_n}_target{target_videos}.log")


    paths = get_paths(0)
    input_path = paths["merged_jsonl"]
    input_path = input_path.replace('.jsonl.gz', f'_top{top_n}.jsonl.gz')

    # Create output path with target count in filename
    dirname = os.path.dirname(input_path)
    basename = os.path.basename(input_path)
    name = basename.replace('.jsonl.gz', '')
    output_path = os.path.join(dirname, f"{name}_aug{target_videos}.jsonl.gz")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    augment_to_target(
        input_path=input_path,
        output_path=output_path,
        target_videos=target_videos
    )

if __name__ == "__main__":
    main()
