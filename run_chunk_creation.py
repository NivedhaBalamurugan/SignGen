import os
from config import *
from utils.jsonl_utils import load_jsonl_gz, save_jsonl_gz

def analyze_chunk(input_filepath, top_n=20):
    """
    Create a new chunk with top N words having most video instances.
    
    Args:
        input_filepath: Path to input chunk file
        top_n: Number of top words to keep
    """
    logging.info(f"Analyzing chunk file: {input_filepath}")
    
    data = load_jsonl_gz(input_filepath)
    if not data:
        logging.error("Failed to load chunk file")
        return None
    
    video_counts = {word: len(videos) for word, videos in data.items()}
    sorted_words = sorted(video_counts.items(), key=lambda x: x[1], reverse=True)
    
    logging.info("\nTop words by video count:")
    for word, count in sorted_words[:top_n]:
        logging.info(f"{word}: {count} videos")
    
    top_words = [word for word, _ in sorted_words[:top_n]]
    
    new_chunk = {word: data[word] for word in top_words}
    
    dirname = os.path.dirname(input_filepath)
    basename = os.path.basename(input_filepath)
    name = basename.replace('.jsonl.gz', '')
    output_filepath = os.path.join(dirname, f"{name}_top{top_n}.jsonl.gz")
    
    save_jsonl_gz(output_filepath, new_chunk, single_object=True)
    
    logging.info(f"\nCreated filtered chunk: {output_filepath}")
    logging.info(f"Original chunk size: {len(data)} words")
    logging.info(f"New chunk size: {len(new_chunk)} words")
    
    total_videos = sum(len(videos) for videos in new_chunk.values())
    logging.info(f"Total videos in new chunk: {total_videos}")
    
    return output_filepath

def main():
    paths = get_paths(0)
    chunk_filepath = paths["merged_jsonl"]
    
    if not os.path.exists(chunk_filepath):
        logging.error(f"Chunk file not found: {chunk_filepath}")
        return
        
    top_n = 50

    analyze_chunk(
        input_filepath=chunk_filepath,
        top_n=top_n
    )

    setup_logging(f"chunk_top_{top_n}.log")

if __name__ == "__main__":
    main()
