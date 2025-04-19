import tensorflow as tf
import numpy as np




INPUT_WORDS = "friend"
CGAN_OUTPUT_FRAMES = "/content/drive/MyDrive/FYP/Models/frames/friend"
CGAN_OUTPUT_VIDEO = "/content/drive/MyDrive/FYP/Models/videos"
# Assuming 'serving_default' is the correct call_endpoint
import tensorflow as tf

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, saved_model_path, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.saved_model = tf.saved_model.load(saved_model_path)

    def call(self, inputs):
        return self.saved_model.signatures['serving_default'](inputs)['output_0']

# Correct the path to point to the 'gen_epoch51' folder:
generator = CustomLayer("/content/drive/MyDrive/FYP/Models/gen_epoch51") 
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

    print(f"Generated sequence for '{word}': {generated_sequence.shape}")
    
    if isSave:
        save_generated_sequence(generated_sequence, CGAN_OUTPUT_FRAMES, CGAN_OUTPUT_VIDEO)

    return generated_sequence
def calculate_diversity_score(real_sequence, generated_sequence):
    """
    Calculate a more stable diversity score between real and generated skeleton sequences.
    """

    if real_sequence.shape[0] != generated_sequence.shape[0]:
        if real_sequence.shape[0] < generated_sequence.shape[0]:
            indices = np.linspace(0, real_sequence.shape[0] - 1, generated_sequence.shape[0], dtype=int)
            real_sequence = real_sequence[indices]
        else:
            indices = np.linspace(0, generated_sequence.shape[0] - 1, real_sequence.shape[0], dtype=int)
            generated_sequence = generated_sequence[indices]

    

    real_range = np.max(real_sequence, axis=0) - np.min(real_sequence, axis=0)
    gen_range = np.max(generated_sequence, axis=0) - np.min(generated_sequence, axis=0)
    range_diff = np.mean(np.abs(real_range - gen_range) / (np.mean(real_range) + 1e-10))
    range_score = np.tanh(range_diff * 3)
    
    real_speed = np.mean(np.abs(np.diff(real_sequence, axis=0)), axis=0)
    gen_speed = np.mean(np.abs(np.diff(generated_sequence, axis=0)), axis=0)
    speed_diff = np.mean(np.abs(real_speed - gen_speed) / (np.mean(real_speed) + 1e-10))
    speed_score = np.tanh(speed_diff * 3)
    
    
    from sklearn.decomposition import PCA
    

    real_flat = real_sequence.reshape(real_sequence.shape[0], -1)
    gen_flat = generated_sequence.reshape(generated_sequence.shape[0], -1)
    
    
    pca_real = PCA(n_components=min(5, real_flat.shape[0], real_flat.shape[1]))
    pca_gen = PCA(n_components=min(5, gen_flat.shape[0], gen_flat.shape[1]))
    
    pca_real.fit(real_flat)
    pca_gen.fit(gen_flat)
    

    real_var = pca_real.explained_variance_ratio_
    gen_var = pca_gen.explained_variance_ratio_
    
    var_diff = np.mean(np.abs(real_var - gen_var))
    pca_score = np.tanh(var_diff * 10)
    

    from scipy import signal
    

    real_psd = np.mean([np.mean(signal.welch(real_sequence[:, i])[1]) for i in range(real_sequence.shape[1])])
    gen_psd = np.mean([np.mean(signal.welch(generated_sequence[:, i])[1]) for i in range(generated_sequence.shape[1])])
    
    psd_diff = np.abs(real_psd - gen_psd) / (real_psd + 1e-10)
    psd_score = np.tanh(psd_diff * 5)
    

    final_score = 0.25 * range_score + 0.25 * speed_score + 0.25 * pca_score + 0.25 * psd_score
    
    
    final_score = max(0, min(1, final_score))
    
    return final_score

def get_diversity_score(input_word, num_samples=5):
    """
    Get a stable diversity score by using fixed seed and averaging multiple metrics.
    """

    real_sequence = get_real_data(input_word)
    

    generated_sequence = generate_skeleton_sequence(input_word, fixed_seed=True)
    
    
    score = calculate_diversity_score(real_sequence, generated_sequence)
    
    print(f"Diversity score for '{input_word}': {score:.4f}")
    return score
if __name__ == "__main__":
  seq= get_cgan_sequence(INPUT_WORDS)
  diversity_score = get_diversity_score(INPUT_WORDS,seq)
    
