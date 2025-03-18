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
from architectures.cgan import *
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
    num_batches = -(-len(jsonl_files)//FILES_PER_BATCH)
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
    checkpoint_dir = os.path.join(os.path.dirname(CGAN_MODEL_PATH), f"checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    gen_path = os.path.join(checkpoint_dir, f"generator_epoch{epoch}_loss{loss:.4f}.keras")
    generator.save(gen_path)
    disc_path = os.path.join(checkpoint_dir, f"discriminator_epoch{epoch}_loss{loss:.4f}.keras")
    discriminator.save(disc_path)
    history_path = os.path.join(checkpoint_dir, f"history_epoch{epoch}.npy")
    np.save(history_path, history)
    logging.info(f"Saved checkpoint for epoch {epoch} with loss {loss:.4f}")
    return gen_path, disc_path
    
def train_gan(generator, discriminator, word_vectors, skeleton_sequences, epochs=100, batch_size=CGAN_BATCH_SIZE, patience=10):
    if not validate_data_shapes(word_vectors, skeleton_sequences):
        return False

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

    total_gen_loss = 0.0
    total_disc_loss = 0.0
    
    # Define joint connections for bone length consistency
    joint_connections = [
        # Upper body connections
        (0, 1),  # Nose to neck
        (1, 2),  # Neck to right shoulder
        (1, 5),  # Neck to left shoulder
        (2, 3),  # Right shoulder to right elbow
        (5, 6),  # Left shoulder to left elbow
        (3, 4),  # Right elbow to right wrist
        (6, 7),  # Left elbow to left wrist
        
        # Left hand connections
        (7, 8),   # Left wrist to left thumb CMC
        (8, 9),   # Left thumb CMC to left thumb tip
        (7, 10),  # Left wrist to left index MCP
        (10, 11), # Left index MCP to left index tip
        (7, 12),  # Left wrist to left middle MCP
        (12, 13), # Left middle MCP to left middle tip
        (7, 14),  # Left wrist to left ring MCP
        (14, 15), # Left ring MCP to left ring tip
        (7, 16),  # Left wrist to left pinky MCP
        (16, 17), # Left pinky MCP to left pinky tip
        
        # Right hand connections
        (4, 18),  # Right wrist to right thumb MCP
        (18, 19), # Right thumb MCP to right thumb tip
        (4, 20),  # Right wrist to right index MCP
        (20, 21), # Right index MCP to right index tip
        (4, 22),  # Right wrist to right middle MCP
        (22, 23), # Right middle MCP to right middle tip
        (4, 24),  # Right wrist to right ring MCP
        (24, 25), # Right ring MCP to right ring tip
        (4, 26),  # Right wrist to right pinky MCP
        (26, 27)  # Right pinky MCP to right pinky tip
    ]

    # Define joint angle limits for anatomical plausibility
    # Format: (joint1, joint2, joint3, min_angle, max_angle)
    # Where joint2 is the vertex of the angle
    joint_angle_limits = [
        # Elbow angles
        (2, 3, 4, 0, 160),   # Right elbow
        (5, 6, 7, 0, 160),   # Left elbow
        
        # Shoulder angles
        (1, 2, 3, 0, 180),   # Right shoulder
        (1, 5, 6, 0, 180),   # Left shoulder
        
        # Wrist angles
        (3, 4, 18, 0, 90),   # Right wrist to thumb base
        (3, 4, 20, 0, 90),   # Right wrist to index base
        (6, 7, 8, 0, 90),    # Left wrist to thumb base
        (6, 7, 10, 0, 90),   # Left wrist to index base
        
        # Left hand finger angles
        (7, 8, 9, 0, 90),    # Left thumb angles
        (7, 10, 11, 0, 90),  # Left index finger
        (7, 12, 13, 0, 90),  # Left middle finger
        (7, 14, 15, 0, 90),  # Left ring finger
        (7, 16, 17, 0, 90),  # Left pinky finger
        
        # Right hand finger angles
        (4, 18, 19, 0, 90),  # Right thumb angles
        (4, 20, 21, 0, 90),  # Right index finger
        (4, 22, 23, 0, 90),  # Right middle finger
        (4, 24, 25, 0, 90),  # Right ring finger
        (4, 26, 27, 0, 90)   # Right pinky finger
    ]

    num_samples = word_vectors.shape[0]
    chunks_per_epoch = max(1, num_samples // MAX_SAMPLES_PER_BATCH + (1 if num_samples % MAX_SAMPLES_PER_BATCH > 0 else 0))

    logging.info(f"Training with {num_samples} samples, {num_samples//batch_size + (1 if num_samples % batch_size > 0 else 0)} batches per epoch")

    best_loss = float('inf')
    patience_counter = 0
    best_generator_weights = None
    best_discriminator_weights = None

    @tf.function
    def train_step(word_vector_batch, real_skeleton_batch):
        actual_batch_size = tf.shape(word_vector_batch)[0]
        noise = tf.random.normal([actual_batch_size, CGAN_NOISE_DIM], dtype=FP_PRECISION)
        word_vector_batch = tf.cast(word_vector_batch, FP_PRECISION)
        generator_input = tf.concat([noise, word_vector_batch], axis=1)
    
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_skeleton = generator(generator_input, training=True)
            real_skeleton_batch = tf.cast(real_skeleton_batch, FP_PRECISION)
            generated_skeleton = tf.cast(generated_skeleton, FP_PRECISION)
            
            mask = create_mask(real_skeleton_batch)
            mask = tf.reduce_mean(mask, axis=1, keepdims=True)
            
            word_vector_expanded = tf.reshape(word_vector_batch, (actual_batch_size, 1, 1, EMBEDDING_DIM))
            word_vector_expanded = tf.tile(word_vector_expanded, [1, 30, 29, 1])
            
            real_input = tf.concat([real_skeleton_batch, word_vector_expanded], axis=-1)
            fake_input = tf.concat([generated_skeleton, word_vector_expanded], axis=-1)
            
            real_output = discriminator(real_input, training=True)
            fake_output = discriminator(fake_input, training=True)
            
            # Basic adversarial losses
            disc_adv_loss = discriminator_loss(real_output, fake_output)
            gen_adv_loss = generator_loss(fake_output)
            
            # Gradient penalty for WGAN-GP
            gp = gradient_penalty(discriminator, real_skeleton_batch, generated_skeleton, word_vector_expanded)
            
            # Additional generator losses
            bone_loss = bone_length_consistency_loss(generated_skeleton, joint_connections)
            motion_loss = motion_smoothness_loss(generated_skeleton)
            anatomical_loss = anatomical_plausibility_loss(generated_skeleton, joint_angle_limits)
            
            # MSE reconstruction loss
            mse_loss = tf.reduce_mean(tf.square(generated_skeleton - real_skeleton_batch))
            
            # Semantic loss to ensure similar words have similar motions
            semantic_loss = semantic_consistency_loss(generated_skeleton, word_vector_batch)
            
            # Combine discriminator loss components
            disc_loss = disc_adv_loss + 10.0 * gp
            
            # Combine generator loss components with meaningful weights
            # The key is to ensure these weights don't lead to components canceling each other
            gen_loss = (
                1.0 * gen_adv_loss +       # Adversarial loss
                10.0 * mse_loss +          # Reconstruction loss (high weight)
                5.0 * bone_loss +          # Bone length consistency
                2.0 * motion_loss +        # Motion smoothness 
                3.0 * anatomical_loss +    # Anatomical plausibility
                2.0 * semantic_loss        # Semantic consistency
            )
            
            # Apply mask to focus on valid joints
            gen_loss = tf.reduce_sum(gen_loss * mask) / tf.reduce_sum(mask)
            disc_loss = tf.reduce_sum(disc_loss * mask) / tf.reduce_sum(mask)
    
        generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        
        # Clip gradients to prevent exploding gradients
        generator_gradients, _ = tf.clip_by_global_norm(generator_gradients, 1.0)
        discriminator_gradients, _ = tf.clip_by_global_norm(discriminator_gradients, 1.0)
        
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
        
        return gen_loss, disc_loss, gen_adv_loss, mse_loss, bone_loss, motion_loss, anatomical_loss, semantic_loss

    def get_model_weights(model):
        return [var.numpy() for var in model.weights]
    
    def set_model_weights(model, weights):
        for i, weight in enumerate(weights):
            model.weights[i].assign(weight)

    for epoch in range(epochs):
        epoch_gen_loss = tf.keras.metrics.Mean()
        epoch_disc_loss = tf.keras.metrics.Mean()
        epoch_adv_loss = tf.keras.metrics.Mean()
        epoch_mse_loss = tf.keras.metrics.Mean()
        epoch_bone_loss = tf.keras.metrics.Mean()
        epoch_motion_loss = tf.keras.metrics.Mean()
        epoch_anatomical_loss = tf.keras.metrics.Mean()
        epoch_semantic_loss = tf.keras.metrics.Mean()

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
                        gen_loss, disc_loss, adv_loss, mse_loss, bone_loss, motion_loss, anatomical_loss, semantic_loss = train_step(word_batch, skeleton_batch)
                        
                        epoch_gen_loss.update_state(gen_loss)
                        epoch_disc_loss.update_state(disc_loss)
                        epoch_adv_loss.update_state(adv_loss)
                        epoch_mse_loss.update_state(mse_loss)
                        epoch_bone_loss.update_state(bone_loss)
                        epoch_motion_loss.update_state(motion_loss)
                        epoch_anatomical_loss.update_state(anatomical_loss)
                        epoch_semantic_loss.update_state(semantic_loss)
                        
                        pbar.set_postfix({
                            'gen_loss': f'{gen_loss:.4f}', 
                            'disc_loss': f'{disc_loss:.4f}'
                        })
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
                        gen_loss, disc_loss, adv_loss, mse_loss, bone_loss, motion_loss, anatomical_loss, semantic_loss = train_step(word_batch, skeleton_batch)
                        
                        epoch_gen_loss.update_state(gen_loss)
                        epoch_disc_loss.update_state(disc_loss)
                        epoch_adv_loss.update_state(adv_loss)
                        epoch_mse_loss.update_state(mse_loss)
                        epoch_bone_loss.update_state(bone_loss)
                        epoch_motion_loss.update_state(motion_loss)
                        epoch_anatomical_loss.update_state(anatomical_loss)
                        epoch_semantic_loss.update_state(semantic_loss)
                        
                        pbar.set_postfix({'gen_loss': f'{gen_loss:.4f}', 'disc_loss': f'{disc_loss:.4f}'})
                        pbar.update(1)
                    except Exception as e:
                        logging.error(f"Error in training partial batch: {e}")

            del word_vectors_chunk, skeleton_sequences_chunk
            gc.collect()

        epoch_gen_loss_value = epoch_gen_loss.result().numpy()
        epoch_disc_loss_value = epoch_disc_loss.result().numpy()
        
        # Better approach to combined loss - use absolute values and weighting
        combined_loss = (
            0.2 * abs(epoch_gen_loss_value) + 
            0.3 * abs(epoch_disc_loss_value) + 
            0.5 * epoch_mse_loss.result().numpy()  # Higher weight on reconstruction quality
        )

        total_gen_loss += epoch_gen_loss_value
        total_disc_loss += epoch_disc_loss_value

        logging.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Gen: {epoch_gen_loss_value:.4f}, "
            f"Disc: {epoch_disc_loss_value:.4f}, "
            f"Combined: {combined_loss:.4f}, "
            f"MSE: {epoch_mse_loss.result().numpy():.4f}, "
            f"Bone: {epoch_bone_loss.result().numpy():.4f}, "
            f"Motion: {epoch_motion_loss.result().numpy():.4f}, "
            f"Anatomical: {epoch_anatomical_loss.result().numpy():.4f}, "
            f"Semantic: {epoch_semantic_loss.result().numpy():.4f}"
        )
        
        # Save checkpoint and history with detailed components
        current_history = {
            'total_gen_loss': total_gen_loss,
            'total_disc_loss': total_disc_loss,
            'epoch': epoch + 1,
            'current_loss': combined_loss,
            'gen_loss': epoch_gen_loss_value,
            'disc_loss': epoch_disc_loss_value,
            'mse_loss': epoch_mse_loss.result().numpy(),
            'bone_loss': epoch_bone_loss.result().numpy(),
            'motion_loss': epoch_motion_loss.result().numpy(),
            'anatomical_loss': epoch_anatomical_loss.result().numpy(),
            'semantic_loss': epoch_semantic_loss.result().numpy()
        }
        
        gen_path, disc_path = save_model_checkpoint(
            generator, 
            discriminator, 
            current_history,
            epoch + 1,
            combined_loss
        )
        
        # Update best model if loss improved
        if combined_loss < best_loss:
            logging.info(f"Loss improved from {best_loss:.4f} to {combined_loss:.4f}")
            best_loss = combined_loss
            patience_counter = 0
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

    os.makedirs(os.path.dirname(CGAN_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CGAN_DIS_PATH), exist_ok=True)

    word_embeddings = load_word_embeddings()
    if not word_embeddings or not validate_word_embeddings(word_embeddings):
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
            save_model_and_history(CGAN_MODEL_PATH, generator, history)
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