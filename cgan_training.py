import os
import glob
import gc
import psutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from config import *
from utils.data_utils import load_skeleton_sequences, load_word_embeddings, prepare_training_data
from utils.validation_utils import validate_data_shapes, validate_config
from utils.model_utils import save_model_and_history, log_model_summary, log_training_config
from utils.glove_utils import validate_word_embeddings
from architectures.cgan import build_generator, build_discriminator, discriminator_loss
from scipy.stats import entropy

FILES_PER_BATCH = 1
MAX_SAMPLES_PER_BATCH = 1000
MEMORY_THRESHOLD = 85

def check_memory():
    memory_percent = psutil.Process().memory_percent()
    if memory_percent > MEMORY_THRESHOLD:
        logging.warning(f"Memory usage high ({memory_percent:.1f}%). Clearing memory...")
        gc.collect()
        return True
    return False

def process_file_batch(files, word_embeddings):
    logging.info(f"Processing files: {[os.path.basename(f) for f in files]}")
    skeleton_data = load_skeleton_sequences(files)
    if not skeleton_data:
        return None, None
    sequences, vectors = prepare_training_data(skeleton_data, word_embeddings)
    del skeleton_data
    gc.collect()
    return sequences, vectors

def process_data_batches(jsonl_files, word_embeddings):
    total_sequences = []
    total_vectors = []
    num_batches = -(-len(jsonl_files)//FILES_PER_BATCH
    with tqdm(total=num_batches, desc="Processing files") as pbar:
        for i in range(0, len(jsonl_files), FILES_PER_BATCH):
            file_batch = jsonl_files[i:i + FILES_PER_BATCH]
            sequences, vectors = process_file_batch(file_batch, word_embeddings)
            if sequences is not None:
                total_sequences.extend(sequences)
                total_vectors.extend(vectors)
            check_memory()
            pbar.update(1)
    return total_sequences, total_vectors

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)  # Clip extreme values

def gradient_penalty(discriminator, real_skeletons, fake_skeletons, word_vectors_expanded):
    """Calculate gradient penalty for WGAN-GP"""
    batch_size = tf.shape(real_skeletons)[0]
    epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    epsilon = tf.tile(epsilon, [1, MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES])
    interpolated = epsilon * real_skeletons + (1 - epsilon) * fake_skeletons
    discriminator_input = tf.concat([interpolated, word_vectors_expanded], axis=-1)
    with tf.GradientTape() as tape:
        tape.watch(discriminator_input)
        pred = discriminator(discriminator_input, training=True)
    gradients = tape.gradient(pred, discriminator_input)
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]) + 1e-8)
    gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
    return gradient_penalty

def calculate_pkd(real_skeletons, generated_skeletons):
    distances = np.linalg.norm(real_skeletons - generated_skeletons, axis=-1)
    avg_distance = np.mean(distances)
    return avg_distance

def calculate_kld(real_data, generated_data):
    real_hist, _ = np.histogram(real_data, bins=30, density=True)
    generated_hist, _ = np.histogram(generated_data, bins=30, density=True)
    kld = entropy(real_hist + 1e-8, generated_hist + 1e-8)
    return kld

def calculate_diversity_score(generated_samples):
    pairwise_distances = []
    for i in range(len(generated_samples)):
        for j in range(i + 1, len(generated_samples)):
            dist = np.linalg.norm(generated_samples[i] - generated_samples[j])
            pairwise_distances.append(dist)
    diversity_score = np.mean(pairwise_distances)
    return diversity_score

def create_mask(real_skeleton_batch):
    mask = tf.reduce_sum(tf.abs(real_skeleton_batch), axis=-1) > 0  
    mask = tf.cast(mask, FP_PRECISION)  
    return mask  

def save_model_checkpoint(generator, discriminator, history, epoch, loss):
    checkpoint_dir = os.path.join(os.path.dirname(CGAN_GEN_PATH), f"checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    gen_path = os.path.join(checkpoint_dir, f"generator_epoch{epoch}_loss{loss:.4f}.keras")
    generator.save(gen_path)
    disc_path = os.path.join(checkpoint_dir, f"discriminator_epoch{epoch}_loss{loss:.4f}.keras")
    discriminator.save(disc_path)
    history_path = os.path.join(checkpoint_dir, f"history_epoch{epoch}.npy")
    np.save(history_path, history)
    logging.info(f"Saved checkpoint for epoch {epoch} with loss {loss:.4f}")
    return gen_path, disc_path
    
def train_gan(generator, discriminator, word_vectors, skeleton_sequences, epochs=100, batch_size=32, patience=10):
    if not validate_data_shapes(word_vectors, skeleton_sequences):
        return False

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

    total_gen_loss = 0.0
    total_disc_loss = 0.0

    num_samples = word_vectors.shape[0]
    chunks_per_epoch = max(1, num_samples // MAX_SAMPLES_PER_BATCH + (1 if num_samples % MAX_SAMPLES_PER_BATCH > 0 else 0))

    logging.info(f"Training with {num_samples} samples, {num_samples//batch_size + (1 if num_samples % batch_size > 0 else 0)} batches per epoch")

    best_loss = float('inf')
    patience_counter = 0
    best_generator_weights = None
    best_discriminator_weights = None
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(word_vector_batch, real_skeleton_batch):
        actual_batch_size = tf.shape(word_vector_batch)[0]
        noise = tf.random.normal([actual_batch_size, CGAN_NOISE_DIM], dtype=FP_PRECISION)
        word_vector_batch = tf.cast(word_vector_batch, FP_PRECISION)
        generator_input = tf.concat([word_vector_batch, noise], axis=1)
    
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_skeleton = generator(generator_input, training=True)
            real_skeleton_batch = tf.cast(real_skeleton_batch, FP_PRECISION)
            generated_skeleton = tf.cast(generated_skeleton, FP_PRECISION)
            mask = create_mask(real_skeleton_batch)  
            mask = tf.reduce_mean(mask, axis=1, keepdims=True)
            word_vector_expanded = tf.reshape(word_vector_batch, (actual_batch_size, 1, 1, 20))
            word_vector_expanded = tf.tile(word_vector_expanded, [1, 30, 29, 1])  # Fixed to 30 frames and 29 joints
            real_input = tf.concat([real_skeleton_batch, word_vector_expanded], axis=-1)
            fake_input = tf.concat([generated_skeleton, word_vector_expanded], axis=-1)
            real_output = discriminator(real_input, training=True)
            fake_output = discriminator(fake_input, training=True)
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
            gp = gradient_penalty(discriminator, real_skeleton_batch, generated_skeleton, word_vector_expanded)
            disc_loss += 10.0 * gp  # Weight for gradient penalty
            gen_loss = tf.reduce_sum(gen_loss * mask) / tf.reduce_sum(mask)
            disc_loss = tf.reduce_sum(disc_loss * mask) / tf.reduce_sum(mask)
    
        generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
        return gen_loss, disc_loss

    def get_model_weights(model):
        return [var.numpy() for var in model.weights]
    
    def set_model_weights(model, weights):
        for i, weight in enumerate(weights):
            model.weights[i].assign(weight)

    for epoch in range(epochs):
        epoch_gen_loss = tf.keras.metrics.Mean()
        epoch_disc_loss = tf.keras.metrics.Mean()

        for chunk_idx in range(chunks_per_epoch):
            chunk_start = chunk_idx * MAX_SAMPLES_PER_BATCH
            chunk_end = min(chunk_start + MAX_SAMPLES_PER_BATCH, num_samples)

            word_vectors_chunk = word_vectors[chunk_start:chunk_end]
            skeleton_sequences_chunk = skeleton_sequences[chunk_start:chunk_end]

            chunk_size = chunk_end - chunk_start
            full_batches = chunk_size // batch_size
            has_partial_batch = chunk_size % batch_size > 0
            chunk_batches = full_batches + (1 if has_partial_batch else 0)

            with tqdm(total=chunk_batches, desc=f"Epoch {epoch+1}/{epochs} Chunk {chunk_idx+1}/{chunks_per_epoch}") as pbar:
                for i in range(full_batches):
                    if check_memory():
                        logging.info("Cleared memory during training")
                    try:
                        batch_start = i * batch_size
                        batch_end = batch_start + batch_size
                        word_batch = word_vectors_chunk[batch_start:batch_end]
                        skeleton_batch = skeleton_sequences_chunk[batch_start:batch_end]
                        gen_loss, disc_loss = train_step(word_batch, skeleton_batch)
                        epoch_gen_loss.update_state(gen_loss)
                        epoch_disc_loss.update_state(disc_loss)
                        pbar.set_postfix({'gen_loss': f'{gen_loss:.4f}', 'disc_loss': f'{disc_loss:.4f}'})
                        pbar.update(1)
                    except Exception as e:
                        logging.error(f"Error in training batch: {e}")
                        continue
                
                if has_partial_batch:
                    if check_memory():
                        logging.info("Cleared memory during training")
                    try:
                        batch_start = full_batches * batch_size
                        word_batch = word_vectors_chunk[batch_start:]
                        skeleton_batch = skeleton_sequences_chunk[batch_start:]
                        gen_loss, disc_loss = train_step(word_batch, skeleton_batch)
                        epoch_gen_loss.update_state(gen_loss)
                        epoch_disc_loss.update_state(disc_loss)
                        pbar.set_postfix({'gen_loss': f'{gen_loss:.4f}', 'disc_loss': f'{disc_loss:.4f}'})
                        pbar.update(1)
                    except Exception as e:
                        logging.error(f"Error in training partial batch: {e}")

            del word_vectors_chunk, skeleton_sequences_chunk
            gc.collect()

        epoch_gen_loss_value = epoch_gen_loss.result().numpy()
        epoch_disc_loss_value = epoch_disc_loss.result().numpy()
        combined_loss = epoch_gen_loss_value + epoch_disc_loss_value
        total_gen_loss += epoch_gen_loss_value
        total_disc_loss += epoch_disc_loss_value

        logging.info(f"Epoch {epoch+1}/{epochs} - Gen Loss: {epoch_gen_loss_value:.4f}, Disc Loss: {epoch_disc_loss_value:.4f}, Combined Loss: {combined_loss:.4f}")
        
        if combined_loss < best_loss:
            logging.info(f"Loss improved from {best_loss:.4f} to {combined_loss:.4f}")
            best_loss = combined_loss
            patience_counter = 0
            current_history = {'total_gen_loss': total_gen_loss, 'total_disc_loss': total_disc_loss, 'epoch': epoch + 1, 'best_loss': best_loss}
            gen_path, disc_path = save_model_checkpoint(generator, discriminator, current_history, epoch + 1, combined_loss)
            best_generator_weights = get_model_weights(generator)
            best_discriminator_weights = get_model_weights(discriminator)
        else:
            patience_counter += 1
            logging.info(f"No improvement in loss. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                if best_generator_weights is not None and best_discriminator_weights is not None:
                    logging.info("Restoring best model weights")
                    set_model_weights(generator, best_generator_weights)
                    set_model_weights(discriminator, best_discriminator_weights)
                break

    logging.info(f"Training complete - Total Generator Loss: {total_gen_loss:.4f}, Total Discriminator Loss: {total_disc_loss:.4f}")
    return True, {'total_gen_loss': total_gen_loss, 'total_disc_loss': total_disc_loss}

def main():
    logging.info("Starting CGAN training process...")
    if not validate_config():
        return

    os.makedirs(os.path.dirname(CGAN_GEN_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CGAN_DIS_PATH), exist_ok=True)

    word_embeddings = load_word_embeddings(GLOVE_TXT_PATH)
    if not word_embeddings or not validate_word_embeddings(word_embeddings, CGAN_NOISE_DIM):
        return

    jsonl_files = sorted(glob.glob(FINAL_JSONL_PATHS))
    if not jsonl_files:
        logging.error(f"No JSONL files found matching pattern: {FINAL_JSONL_PATHS}")
        return

    total_sequences = []
    total_vectors = []

    for i in range(0, len(jsonl_files), FILES_PER_BATCH):
        file_batch = jsonl_files[i:i + FILES_PER_BATCH]
        logging.info(f"Processing batch {i//FILES_PER_BATCH + 1}/{-(-len(jsonl_files)//FILES_PER_BATCH)}")
        sequences, vectors = process_file_batch(file_batch, word_embeddings)
        if sequences is not None:
            total_sequences.extend(sequences)
            total_vectors.extend(vectors)
        check_memory()

    if not total_sequences:
        logging.error("No valid sequences loaded")
        return

    all_skeleton_sequences = np.array(total_sequences)
    all_word_vectors = np.array(total_vectors)

    del total_sequences, total_vectors, word_embeddings
    gc.collect()

    generator = build_generator()
    discriminator = build_discriminator()

    log_model_summary(generator, "Generator")
    log_model_summary(discriminator, "Discriminator")
    log_training_config()

    try:
        success, history = train_gan(generator, discriminator, all_word_vectors, all_skeleton_sequences, epochs=CGAN_EPOCHS, batch_size=CGAN_BATCH_SIZE, patience=20)
        if success:
            save_model_and_history(CGAN_GEN_PATH, generator, history)
            save_model_and_history(CGAN_DIS_PATH, discriminator)
            generated_skeletons = generator(all_word_vectors)
            real_skeletons = all_skeleton_sequences
            pkd_score = calculate_pkd(real_skeletons, generated_skeletons)
            kld_score = calculate_kld(real_skeletons.flatten(), generated_skeletons.flatten())
            diversity_score = calculate_diversity_score(generated_skeletons)
            logging.info(f"Per-Keypoint Distance (PKD): {pkd_score:.4f}")
            logging.info(f"KL Divergence (KLD): {kld_score:.4f}")
            logging.info(f"Diversity Score: {diversity_score:.4f}")
        else:
            logging.error("Training failed, models not saved")
    except Exception as e:
        logging.error(f"Training error: {e}")
        return

if __name__ == "__main__":
    setup_logging("cgan_training")
    main()
