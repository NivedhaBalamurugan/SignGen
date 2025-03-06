import os
import json
import numpy as np
from collections import defaultdict
from augmentation import (generate_params, 
                        augment_skeleton_sequence_combined)
from utils.jsonl_utils import load_jsonl_gz, save_jsonl_gz
from utils.data_utils import (denormalize_landmarks, normalize_landmarks)
from config import *
import glob

def augment_dataset(input_path, output_path, target_videos=100):
    """Augment a single dataset to have target number of videos with consistent augmentation parameters."""
    try:
        data = load_jsonl_gz(input_path)
        logging.info(f"Loaded {len(data)} words from {input_path}")
    except Exception as e:
        logging.error(f"Error loading JSONL file: {e}")
        return None, None

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
                        
                    # Generate parameters ONCE for the entire video
                    shear_params, trans_params, scale_params = generate_params()
                    
                    # Select augmentation method ONCE for the entire video
                    new_video = []
                    for frame in video:
                        # Denormalize landmarks
                        frame = np.array(frame, dtype=np.float32)
                        frame = denormalize_landmarks(frame)
                        
                        # Apply combined augmentations
                        combined_augmentations = augment_skeleton_sequence_combined(
                            frame, 
                            shear_params, 
                            trans_params, 
                            scale_params
                        )
                        
                        # Randomly select one augmentation method
                        aug_method = np.random.choice(list(combined_augmentations.keys()))
                        augmented_frame = combined_augmentations[aug_method]
                        
                        # Normalize and add to video
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

        # Prepare stats for this specific dataset
        dataset_stats = {
            'dataset_name': os.path.basename(input_path),
            'total_words': len(data),
            'total_original_videos': total_original,
            'total_augmented_videos': total_augmented,
            'total_videos': total_original + total_augmented,
            'target_per_word': target_videos
        }
        
        # Save augmented data
        save_jsonl_gz(output_path, augmented_data, single_object=True)
        
        logging.info(f"Successfully augmented dataset:")
        logging.info(f"Original videos: {total_original}")
        logging.info(f"Augmented videos: {total_augmented}")
        logging.info(f"Total videos: {total_original + total_augmented}")
        logging.info(f"Saved to: {output_path}")

        return augmented_data, dataset_stats

    except Exception as e:
        logging.error(f"Error during augmentation: {e}")
        return None, None

def main():
    #top_n = 20
    target_videos = 6000

    # Setup logging
    setup_logging(f"split_chunk_augmentation.log")

    # Get paths
    paths = get_paths(0)
    input_dir = os.path.dirname(MERGED_PATH)
    
    # Find all input files matching the pattern
    input_pattern = f"{MERGED_PATH}/splits/*.jsonl.gz"
    input_files = glob.glob(input_pattern)
    
    if not input_files:
        logging.error(f"No input files found matching pattern: {input_pattern}")
        return

    # Prepare output directory
    output_dir = os.path.join(input_dir, f"split_augmentation")
    os.makedirs(output_dir, exist_ok=True)

    # Comprehensive stats to collect across all datasets
    comprehensive_stats = {
        'total_datasets': len(input_files),
        'datasets': {}
    }

    # Process each input file
    for input_path in input_files:
        # Generate output path for this specific file
        basename = os.path.basename(input_path)
        name = basename.replace('.jsonl.gz', '')
        output_path = os.path.join(output_dir, f"{name}_aug.jsonl.gz")
        
        # Perform augmentation
        augmented_data, dataset_stats = augment_dataset(
            input_path=input_path,
            output_path=output_path,
            target_videos=target_videos
        )
        
        # Collect stats if augmentation was successful
        if dataset_stats:
            comprehensive_stats['datasets'][dataset_stats['dataset_name']] = dataset_stats

    # Update comprehensive stats with summary
    comprehensive_stats['total_words'] = sum(
        stats['total_words'] for stats in comprehensive_stats['datasets'].values()
    )
    comprehensive_stats['total_original_videos'] = sum(
        stats['total_original_videos'] for stats in comprehensive_stats['datasets'].values()
    )
    comprehensive_stats['total_augmented_videos'] = sum(
        stats['total_augmented_videos'] for stats in comprehensive_stats['datasets'].values()
    )
    comprehensive_stats['total_videos'] = sum(
        stats['total_videos'] for stats in comprehensive_stats['datasets'].values()
    )

    # Save comprehensive stats
    stats_path = os.path.join(output_dir, f"augmentation_split_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(comprehensive_stats, f, indent=2)

    logging.info(f"Augmentation completed. Stats saved to: {stats_path}")

if __name__ == "__main__":
    main()