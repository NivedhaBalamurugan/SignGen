import os
import json
import tensorflow as tf
import numpy as np
import show_output
from config import *
from utils.data_utils import convert_line_segments_to_skeleton, load_word_embeddings

INPUT_WORD = "book"

generator = tf.keras.models.load_model("generator_epoch63_loss0.0385.keras")


def generate_skeleton_sequence(word):
    word_embeddings = load_word_embeddings()
    if word not in word_embeddings:
        print(f"Word '{word}' not found in embeddings.")
        return None

    word_vector = np.array(word_embeddings[word], dtype=np.float32).reshape(1, -1)
    noise = tf.random.normal([1, CGAN_NOISE_DIM], dtype=tf.float32)
    generator_input = tf.concat([noise, word_vector], axis=1)
    
    generated_skeleton = generator(generator_input, training=False).numpy()
    return generated_skeleton.squeeze()

def get_cgan_sequence(word, isSave_Video):
    generated_sequence = generate_skeleton_sequence(word)
    if generated_sequence is not None:
        print(f"Generated Skeleton Shape (line segments) for '{word}': {generated_sequence.shape}")
        line_segment_data = {word: generated_sequence.reshape(1, *generated_sequence.shape)}
        skeleton_data = convert_line_segments_to_skeleton(line_segment_data)
        skeleton_sequence = skeleton_data[word].squeeze()
        
        save_path = OUTPUTS_PATH
        os.makedirs(save_path, exist_ok=True)
        json_filepath = os.path.join(save_path, f"generated_skeleton_{word}.json")
        with open(json_filepath, 'w') as json_file:
            json.dump(skeleton_sequence.tolist(), json_file)
            print("Saved successfully")
            
        if isSave_Video:
            show_output.save_generated_sequence(skeleton_sequence, CGAN_OUTPUT_FRAMES, CGAN_OUTPUT_VIDEO) 
    return skeleton_sequence

if __name__ == "__main__":
    get_cgan_sequence(INPUT_WORD,1)