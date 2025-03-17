import time
import os
import json
import numpy as np
from collections import defaultdict
from augmentation import (generate_params, 
                        augment_skeleton_sequence_combined)
from utils.jsonl_utils import load_jsonl_gz, save_jsonl_gz
from utils.data_utils import (denormalize_landmarks, normalize_landmarks, select_sign_frames)
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
        running_stats = {
            'total_videos_processed': 0,
            'total_successful_augmentations': 0,
            'total_failed_augmentations': 0
        }

        for word, videos in data.items():
            original_count = len(videos)
            total_original += original_count
            
            # Add original videos to output
            augmented_data[word].extend(videos)
            
            # Calculate how many copies we need
            augmentations_needed = max(0, target_videos - original_count)
            copies_per_video = -(-augmentations_needed // original_count) if original_count > 0 else 0
            
            logging.info(f"Processing '{word}': {original_count} original videos, need {augmentations_needed} more")
            
            # Examine the structure of first video for debugging
            if original_count > 0:
                first_video = videos[0]
                logging.info(f"First video for '{word}' - Type: {type(first_video)}, Length: {len(first_video) if isinstance(first_video, (list, tuple)) else 'N/A'}")
                
                if isinstance(first_video, list) and len(first_video) > 0:
                    first_frame = first_video[0]
                    logging.info(f"First frame - Type: {type(first_frame)}, Shape/Length: {first_frame.shape if hasattr(first_frame, 'shape') else len(first_frame) if isinstance(first_frame, (list, tuple)) else 'N/A'}")
                    
                    # If first_frame is a list and not empty, check its first element
                    if isinstance(first_frame, list) and len(first_frame) > 0:
                        first_landmark = first_frame[0]
                        logging.info(f"First landmark - Type: {type(first_landmark)}, Value: {first_landmark}")
            
            # Skip further processing if no videos need augmentation
            if augmentations_needed <= 0:
                logging.info(f"No augmentation needed for '{word}', already has {original_count} videos")
                stats[word] = {
                    'original_videos': original_count,
                    'augmented_videos': 0,
                    'total_videos': original_count
                }
                continue
            
            # Keep track of successful augmentations
            successful_augmentations = 0
            
            for video_idx, video in enumerate(videos, 1):
                running_stats['total_videos_processed'] += 1
                
                for copy in range(copies_per_video):
                    if len(augmented_data[word]) >= target_videos:
                        break
                    
                    try:
                        # Try to get frames for augmentation
                        if len(video) < 30:
                            continue
                        video_frames = select_sign_frames(video)
                        if len(video_frames) < 30:
                            continue
                        # Generate parameters ONCE for the entire video
                        shear_params, trans_params, scale_params = generate_params()
                        
                        # Create a new augmented video
                        new_video = []
                        
                        for frame in video_frames:
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
                        successful_augmentations += 1
                        total_augmented += 1
                        running_stats['total_successful_augmentations'] += 1
                        
                        if video_idx % 10 == 0 or successful_augmentations % 100 == 0:
                            logging.info(f"Created {len(augmented_data[word])}/{target_videos} videos for '{word}'")
                            logging.info(f"Running stats: {running_stats}")
                    
                    except ValueError as e:
                        # Log specific value errors from select_sign_frames
                        if video_idx <= 10:  # Only log first few to avoid cluttering logs
                            logging.warning(f"Could not select sign frames for '{word}' video {video_idx}: {e}")
                    except Exception as e:
                        running_stats['total_failed_augmentations'] += 1
                        if video_idx <= 10:  # Only log first few to avoid cluttering logs
                            logging.warning(f"Error during augmentation for '{word}' video {video_idx}: {e}")

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
            'target_per_word': target_videos,
            'word_stats': stats,
            'running_stats': running_stats
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
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None, None
    
def main():
    #top_n = 20
    target_videos = 6000

    # Setup logging
    setup_logging(f"split_chunk_augmentation_{int(time.time())}.log")

    # Get paths
    paths = get_paths(0)
    input_dir = os.path.dirname(MERGED_PATH)
    
    # Find all input files matching the pattern
    input_pattern = f"Dataset\\new_landmarks\splits\*.jsonl.gz"
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

    # Create word-level stats
    word_stats = {}
    for dataset_name, dataset in comprehensive_stats['datasets'].items():
        if 'word_stats' in dataset:
            for word, stats in dataset['word_stats'].items():
                if word not in word_stats:
                    word_stats[word] = {
                        'original_videos': 0,
                        'augmented_videos': 0,
                        'total_videos': 0
                    }
                word_stats[word]['original_videos'] += stats['original_videos']
                word_stats[word]['augmented_videos'] += stats['augmented_videos']
                word_stats[word]['total_videos'] += stats['total_videos']

    # Save word-level stats
    word_stats_path = os.path.join(output_dir, f"word_level_stats.json")
    with open(word_stats_path, 'w') as f:
        json.dump(word_stats, f, indent=2)

    # Save comprehensive stats
    stats_path = os.path.join(output_dir, f"augmentation_split_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(comprehensive_stats, f, indent=2)

    logging.info(f"Augmentation completed. Stats saved to: {stats_path}")

if __name__ == "__main__":
    main()