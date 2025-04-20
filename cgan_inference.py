import tensorflow as tf
import numpy as np


INPUT_WORDS = "star"
CGAN_OUTPUT_FRAMES = "/content/drive/MyDrive/FYP/Models/frames_cpu/star"
CGAN_OUTPUT_VIDEO = "/content/drive/MyDrive/FYP/Models/videos"
generator = tf.keras.models.load_model("/content/drive/MyDrive/FYP/Models/cgan_shuffle+old_archi_30_epochs/generator_epoch30_loss0.1639.keras")

def generate_skeleton_sequence(word, fixed_seed=True):
    word_embeddings = load_word_embeddings()
    word_vector = np.array(word_embeddings[word], dtype=np.float32).reshape(1, -1)

    if fixed_seed:

        word_hash = hash(word) % 10000
        tf.random.set_seed(word_hash)
        np.random.seed(word_hash)
    
    noise = tf.random.normal([1, CGAN_NOISE_DIM], dtype=tf.float32)
    generator_input = tf.concat([noise, word_vector], axis=1)
    
    generated_skeleton = generator(generator_input).numpy()
    return generated_skeleton.squeeze()

def get_cgan_sequence(word, isSave=True):
    generated_sequence = generate_skeleton_sequence(word)
    if generated_sequence is None:
        return None

    print(f"Generated CGAN sequence for '{word}': {generated_sequence.shape}")
    
    if isSave:
        save_generated_sequence(generated_sequence, CGAN_OUTPUT_FRAMES, CGAN_OUTPUT_VIDEO)

    

    return generated_sequence
def read_real_data(file_path):
    with open(file_path, 'r') as file:
        real_data = json.load(file)

    for gloss, data in real_data.items():
        real_data[gloss] = np.array(data, dtype=np.float32)
    
    return real_data

def get_real_data(gloss):
    
    real_data_path = "/content/drive/MyDrive/FYP/Models/classifier/real_data.json"
    real_data = read_real_data(real_data_path)
    return real_data[gloss]
def calculate_diversity_score(real_sequence, generated_sequence):
    """
    Calculate diversity score between real and generated skeleton sequences.
    Returns a score between 0 and 1, where:
    - 0 means sequences are identical
    - 1 means sequences are completely different
    """
    # Make sure sequences have the same shape
    # If different lengths, we'll resize the shorter one
    if real_sequence.shape[0] != generated_sequence.shape[0]:
        # Determine which sequence is shorter
        if real_sequence.shape[0] < generated_sequence.shape[0]:
            # Resize real_sequence to match generated_sequence
            indices = np.linspace(0, real_sequence.shape[0] - 1, generated_sequence.shape[0], dtype=int)
            real_sequence = real_sequence[indices]
        else:
            # Resize generated_sequence to match real_sequence
            indices = np.linspace(0, generated_sequence.shape[0] - 1, real_sequence.shape[0], dtype=int)
            generated_sequence = generated_sequence[indices]
    
    # Normalize both sequences to have zero mean and unit variance
    # This focuses on the pattern rather than absolute values
    real_normalized = (real_sequence - np.mean(real_sequence, axis=0)) / (np.std(real_sequence, axis=0) + 1e-10)
    gen_normalized = (generated_sequence - np.mean(generated_sequence, axis=0)) / (np.std(generated_sequence, axis=0) + 1e-10)
    
    # Calculate various metrics
    
    # 1. Dynamic Time Warping distance (approximated with MSE)
    mse = np.mean((real_normalized - gen_normalized) ** 2)
    mse_score = np.tanh(mse)  # Convert to 0-1 range
    
    # 2. Correlation difference
    real_corr = np.corrcoef(real_normalized.reshape(real_normalized.shape[0], -1).T)
    gen_corr = np.corrcoef(gen_normalized.reshape(gen_normalized.shape[0], -1).T)
    corr_diff = np.mean(np.abs(real_corr - gen_corr))
    corr_score = np.tanh(corr_diff * 5)  # Scale and convert to 0-1 range
    
    # 3. Fourier transform difference (captures rhythm differences)
    real_fft = np.abs(np.fft.fft(real_normalized, axis=0))
    gen_fft = np.abs(np.fft.fft(gen_normalized, axis=0))
    fft_diff = np.mean(np.abs(real_fft - gen_fft))
    fft_score = np.tanh(fft_diff)
    
    # Combine scores with weights
    final_score = 0.4 * mse_score + 0.3 * corr_score + 0.3 * fft_score
    
    # Ensure score is in 0-1 range
    final_score = max(0, min(1, final_score))
    
    return final_score

def get_diversity_score(input_word,seq):
    """
    Compare generated sequence with real sequence for the input word
    and return a diversity score between 0 and 1.
    """
    # Get real skeleton sequence
    real_sequence = get_real_data(input_word)
    
    # Get generated skeleton sequence
    #generated_sequence = get_cgan_sequence(input_word, isSave=False)
    
    # Calculate diversity score
    score = calculate_diversity_score(real_sequence, seq)
    
    print(f"Diversity score for '{input_word}': {score:.4f}")
    return score

if __name__ == "__main__":
  seq= get_cgan_sequence(INPUT_WORDS)
  diversity_score = get_diversity_score(INPUT_WORDS,seq)
