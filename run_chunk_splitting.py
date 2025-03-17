import os
from utils.jsonl_utils import load_jsonl_gz, save_jsonl_gz
from config import *


WORDS_PER_SPLIT = 1

def split_chunk_by_words(input_path, output_dir, words_per_split=WORDS_PER_SPLIT):
    """Split a chunk file into multiple files with equal number of words."""
    
    logging.info(f"Loading data from: {input_path}")
    data = load_jsonl_gz(input_path)
    if not data:
        logging.error("Failed to load input file")
        return
    
    words = list(data.keys())
    total_words = len(words)
    num_files = -(-total_words // words_per_split)
    
    logging.info(f"Splitting {total_words} words into {num_files} files ({words_per_split} words per file)")
    os.makedirs(output_dir, exist_ok=True)
    
    for file_idx in range(num_files):
        start_idx = file_idx * words_per_split
        end_idx = min((file_idx + 1) * words_per_split, total_words)
        
        # Get current batch of words
        current_words = words[start_idx:end_idx]
        subset_data = {word: data[word] for word in current_words}
        
        # Generate output path
        basename = os.path.basename(input_path).replace('.jsonl.gz', '')
        output_path = os.path.join(
            output_dir, 
            f"{basename}_split{file_idx+1}.jsonl.gz"
        )
        
        # Save split file
        save_jsonl_gz(output_path, subset_data, single_object=True)
        
        # Log info
        total_videos = sum(len(videos) for videos in subset_data.values())
        logging.info(f"\nSplit {file_idx+1} of {num_files}:")
        logging.info(f"Output: {output_path}")
        logging.info(f"Words ({len(current_words)}): {', '.join(current_words)}")
        logging.info(f"Total videos: {total_videos}")

def main():
    # Get paths from config
    paths = get_paths(0)
    input_path = f"Dataset\\new_landmarks\extracted_merged.jsonl.gz"
    
    # Create output directory next to input file
    output_dir = os.path.join(os.path.dirname(input_path), 'splits')
    
    # Setup logging
    # setup_logging("chunk_splitting")
    
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return
    
    split_chunk_by_words(
        input_path=input_path,
        output_dir=output_dir,
        words_per_split=WORDS_PER_SPLIT
    )

if __name__ == "__main__":
    main()
