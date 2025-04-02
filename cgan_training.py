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

MODEL_NAME = "shuffle_remove_batching"

FILES_PER_BATCH = 1
MAX_SAMPLES_PER_BATCH = 1000

def process_file_batch_with_labels(files, word_embeddings):
    logging.info(f"Processing files with labels: {[os.path.basename(f) for f in files]}")
    
    # Load skeleton data using the existing function
    skeleton_data = load_skeleton_sequences(files)
    if not skeleton_data:
        return None, None, None
    
    # Get sequences and vectors using the existing function
    sequences, vectors = prepare_training_data(skeleton_data, word_embeddings)
    
    # Create word labels (mapping words to indices)
    word_to_class = {}
    class_idx = 0
    words = list(skeleton_data.keys())
    
    for word in words:
        if word not in word_to_class and word in word_embeddings:
            word_to_class[word] = class_idx
            class_idx += 1
    
    # Create the labels array based on the order of sequences
    labels = []
    current_index = 0
    for word in words:
        if word in word_embeddings:
            num_sequences = len(skeleton_data[word])
            labels.extend([word_to_class[word]] * num_sequences)
    
    labels = np.array(labels)
    
    logging.info(f"Processed {len(sequences)} sequences from {len(files)} files")
    logging.info(f"Found {len(word_to_class)} unique words assigned to classes")
    
    # Clean up to free memory
    del skeleton_data
    gc.collect()
    
    return sequences, vectors, labels

def save_model_checkpoint(generator, discriminator, history, epoch, loss):
    checkpoint_dir = os.path.join(os.path.dirname(CGAN_MODEL_PATH), f"checkpoints_{MODEL_NAME}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    gen_path = os.path.join(checkpoint_dir, f"generator_epoch{epoch}_loss{loss:.4f}.keras")
    generator.save(gen_path)
    disc_path = os.path.join(checkpoint_dir, f"discriminator_epoch{epoch}_loss{loss:.4f}.keras")
    discriminator.save(disc_path)
    history_path = os.path.join(checkpoint_dir, f"history_epoch{epoch}.npy")
    np.save(history_path, history)
    logging.info(f"Saved checkpoint for epoch {epoch} with loss {loss:.4f}")
    return gen_path, disc_path

def enhanced_gradient_penalty(discriminator, real_skeletons, fake_skeletons, word_vectors_expanded):
    """Calculate improved gradient penalty for WGAN-GP"""
    batch_size = tf.shape(real_skeletons)[0]
    
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    differences = fake_skeletons - real_skeletons
    interpolates = real_skeletons + (alpha * differences)
    
    feature_noise = tf.random.normal(tf.shape(interpolates), mean=0.0, stddev=0.01)
    interpolates_noisy = interpolates + feature_noise
    
    discriminator_input = tf.concat([interpolates_noisy, word_vectors_expanded], axis=-1)
    
    with tf.GradientTape() as tape:
        tape.watch(discriminator_input)
        pred = discriminator(discriminator_input, training=True)
    
    gradients = tape.gradient(pred, discriminator_input)
    
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=[1, 2, 3])
    gradient_l2_norm = tf.sqrt(gradients_sqr_sum + 1e-8)
    
    gradient_penalty = tf.reduce_mean(tf.square(gradient_l2_norm - 1.0))
    
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

def shuffle_training_data(word_vectors, skeleton_sequences, word_labels):
    assert len(word_vectors) == len(skeleton_sequences) == len(word_labels)
    indices = np.arange(len(word_vectors))
    np.random.shuffle(indices)
    return (
        word_vectors[indices], 
        skeleton_sequences[indices], 
        word_labels[indices]
    )
    
def train_gan(generator, discriminator, word_vectors, skeleton_sequences, word_labels, epochs=100, batch_size=CGAN_BATCH_SIZE, patience=10):
    if not validate_data_shapes(word_vectors, skeleton_sequences):
        return False

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

    total_gen_loss = 0.0
    total_disc_loss = 0.0

    num_samples = word_vectors.shape[0]
    
    logging.info(f"Training with {num_samples} samples, {num_samples//batch_size + (1 if num_samples % batch_size > 0 else 0)} batches per epoch")

    best_loss = float('inf')
    patience_counter = 0
    best_generator_weights = None
    best_discriminator_weights = None
    
    all_diversity_scores = []
    word_specific_outputs = {i: [] for i in range(20)}  # Assuming 20 word classes

    @tf.function
    def train_step(word_vector_batch, real_skeleton_batch, word_label_batch):
        actual_batch_size = tf.shape(word_vector_batch)[0]
        noise = tf.random.normal([actual_batch_size, CGAN_NOISE_DIM], dtype=FP_PRECISION)
        word_vector_batch = tf.cast(word_vector_batch, FP_PRECISION)
        
        # Add noise to word embeddings for robustness
        word_vector_batch_noisy = word_vector_batch + tf.random.normal(
            tf.shape(word_vector_batch), mean=0.0, stddev=0.02, dtype=FP_PRECISION
        )
        
        generator_input = tf.concat([noise, word_vector_batch_noisy], axis=1)
    
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
            
            # Access the full model to get both outputs
            real_output, real_class_output = discriminator.full_model(real_input, training=True)
            fake_output, fake_class_output = discriminator.full_model(fake_input, training=True)
            
            # Basic adversarial losses (WGAN)
            disc_adv_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            gen_adv_loss = -tf.reduce_mean(fake_output)
            
            # Classification loss for discriminator (only on real samples)
            real_class_loss = tf.keras.losses.sparse_categorical_crossentropy(
                word_label_batch, real_class_output)
            real_class_loss = tf.reduce_mean(real_class_loss)
            
            # Classification loss for generator (on fake samples)
            fake_class_loss = tf.keras.losses.sparse_categorical_crossentropy(
                word_label_batch, fake_class_output)
            fake_class_loss = tf.reduce_mean(fake_class_loss)
            
            # Gradient penalty for WGAN-GP
            gp = enhanced_gradient_penalty(discriminator, real_skeleton_batch, generated_skeleton, word_vector_expanded)
            
            # Additional generator losses
            bone_loss = bone_length_consistency_loss(generated_skeleton)
            motion_loss = motion_smoothness_loss(generated_skeleton)
            anatomical_loss = anatomical_plausibility_loss(generated_skeleton)
            
            # MSE reconstruction loss
            mse_loss = tf.reduce_mean(tf.square(generated_skeleton - real_skeleton_batch))
            
            # Semantic loss to ensure similar words have similar motions
            semantic_loss = semantic_consistency_loss(generated_skeleton, word_vector_batch)
            
            # Combine discriminator loss components
            disc_loss = disc_adv_loss + 10.0 * gp + 2.0 * real_class_loss
            
            # Combine generator loss components with meaningful weights
            # Increased weight on classification loss to help with mode collapse
            gen_loss = (
                1.0 * gen_adv_loss +       # Adversarial loss
                10.0 * mse_loss +          # Reconstruction loss (high weight)
                5.0 * bone_loss +          # Bone length consistency
                2.0 * motion_loss +        # Motion smoothness 
                3.0 * anatomical_loss +    # Anatomical plausibility
                2.0 * semantic_loss +      # Semantic consistency
                5.0 * fake_class_loss      # Classification loss
            )
            
            # Apply mask to focus on valid joints
            gen_loss = tf.reduce_sum(gen_loss * mask) / tf.reduce_sum(mask)
            disc_loss = tf.reduce_sum(disc_loss * mask) / tf.reduce_sum(mask)
    
        generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.full_model.trainable_variables)
        
        # Clip gradients to prevent exploding gradients
        generator_gradients, _ = tf.clip_by_global_norm(generator_gradients, 1.0)
        discriminator_gradients, _ = tf.clip_by_global_norm(discriminator_gradients, 1.0)
        
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.full_model.trainable_variables))
        
        return gen_loss, disc_loss, gen_adv_loss, fake_class_loss, mse_loss, bone_loss, motion_loss, anatomical_loss, semantic_loss

    def get_model_weights(model):
        return [var.numpy() for var in model.weights]
    
    def set_model_weights(model, weights):
        for i, weight in enumerate(weights):
            model.weights[i].assign(weight)
            
    # Helper functions for diversity tracking
    def get_representative_word_vectors():
        # Return one representative vector for each word class (20 classes assumed)
        representative_vectors = []
        for class_idx in range(20):
            # Find all samples with this label
            class_indices = np.where(word_labels == class_idx)[0]
            if len(class_indices) > 0:
                # Use the mean vector as representative
                mean_vector = np.mean(word_vectors[class_indices], axis=0)
                representative_vectors.append(mean_vector)
            else:
                # Fallback if no samples for this class
                representative_vectors.append(np.zeros(EMBEDDING_DIM))
        return np.array(representative_vectors)
    
    def get_diverse_word_vectors(count):
        # Sample diverse word vectors for evaluation
        if count <= len(word_vectors):
            # Randomly sample from available vectors
            indices = np.random.choice(len(word_vectors), count, replace=False)
            return word_vectors[indices]
        else:
            # If requesting more than available, reuse with some noise
            indices = np.random.choice(len(word_vectors), count, replace=True)
            vectors = word_vectors[indices]
            # Add small noise for diversity
            noise = np.random.normal(0, 0.05, vectors.shape)
            return vectors + noise

    for epoch in range(epochs):
        # Shuffle data at the start of each epoch
        word_vectors_shuffled, skeleton_sequences_shuffled, word_labels_shuffled = shuffle_training_data(
            word_vectors, skeleton_sequences, word_labels
        )
        
        epoch_gen_loss = tf.keras.metrics.Mean()
        epoch_disc_loss = tf.keras.metrics.Mean()
        epoch_adv_loss = tf.keras.metrics.Mean()
        epoch_class_loss = tf.keras.metrics.Mean()
        epoch_mse_loss = tf.keras.metrics.Mean()
        epoch_bone_loss = tf.keras.metrics.Mean()
        epoch_motion_loss = tf.keras.metrics.Mean()
        epoch_anatomical_loss = tf.keras.metrics.Mean()
        epoch_semantic_loss = tf.keras.metrics.Mean()

        # Calculate total batches for the epoch
        num_samples = word_vectors_shuffled.shape[0]
        batches_per_epoch = num_samples // batch_size + (1 if num_samples % batch_size > 0 else 0)
        
        with tqdm(total=batches_per_epoch, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i in range(batches_per_epoch):
                batch_start = i * batch_size
                batch_end = min(batch_start + batch_size, num_samples)
                
                word_batch = word_vectors_shuffled[batch_start:batch_end]
                skeleton_batch = skeleton_sequences_shuffled[batch_start:batch_end]
                label_batch = word_labels_shuffled[batch_start:batch_end]
                
                try:
                    gen_loss, disc_loss, adv_loss, class_loss, mse_loss, bone_loss, motion_loss, anatomical_loss, semantic_loss = train_step(
                        word_batch, skeleton_batch, label_batch
                    )
                    
                    epoch_gen_loss.update_state(gen_loss)
                    epoch_disc_loss.update_state(disc_loss)
                    epoch_adv_loss.update_state(adv_loss)
                    epoch_class_loss.update_state(class_loss)
                    epoch_mse_loss.update_state(mse_loss)
                    epoch_bone_loss.update_state(bone_loss)
                    epoch_motion_loss.update_state(motion_loss)
                    epoch_anatomical_loss.update_state(anatomical_loss)
                    epoch_semantic_loss.update_state(semantic_loss)
                    
                    pbar.set_postfix({
                        'gen_loss': f'{gen_loss:.4f}', 
                        'disc_loss': f'{disc_loss:.4f}',
                        'class_loss': f'{class_loss:.4f}'
                    })
                    pbar.update(1)
                except Exception as e:
                    logging.error(f"Error in training batch: {e}")
                    continue
            
        # At the end of each epoch, generate samples for each word and track diversity
        test_noise = tf.random.normal([20, CGAN_NOISE_DIM])  # One sample per word class
        test_word_vectors = get_representative_word_vectors()  # Get one vector per word class
        
        # Concatenate noise with word vectors
        test_inputs = tf.concat([test_noise, test_word_vectors], axis=1)
        
        # Generate samples
        test_samples = generator(test_inputs)
        
        # Calculate diversity between samples from different words
        diversity_score = calculate_diversity_score(test_samples.numpy())
        all_diversity_scores.append(diversity_score)
        
        # Store samples for each word
        for i in range(20):
            word_specific_outputs[i].append(test_samples[i].numpy())

        epoch_gen_loss_value = epoch_gen_loss.result().numpy()
        epoch_disc_loss_value = epoch_disc_loss.result().numpy()
        epoch_class_loss_value = epoch_class_loss.result().numpy()
        
        # Better approach to combined loss - use absolute values and weighting
        combined_loss = (
            0.2 * abs(epoch_gen_loss_value) + 
            0.3 * abs(epoch_disc_loss_value) + 
            0.2 * epoch_class_loss_value +
            0.5 * epoch_mse_loss.result().numpy()  # Higher weight on reconstruction quality
        )

        total_gen_loss += epoch_gen_loss_value
        total_disc_loss += epoch_disc_loss_value

        logging.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Gen: {epoch_gen_loss_value:.4f}, "
            f"Disc: {epoch_disc_loss_value:.4f}, "
            f"Class: {epoch_class_loss_value:.4f}, "
            f"Combined: {combined_loss:.4f}, "
            f"MSE: {epoch_mse_loss.result().numpy():.4f}, "
            f"Diversity: {diversity_score:.4f}"
        )
        
        # Save checkpoint and history with detailed components
        current_history = {
            'total_gen_loss': total_gen_loss,
            'total_disc_loss': total_disc_loss,
            'epoch': epoch + 1,
            'current_loss': combined_loss,
            'gen_loss': epoch_gen_loss_value,
            'disc_loss': epoch_disc_loss_value,
            'class_loss': epoch_class_loss_value,
            'mse_loss': epoch_mse_loss.result().numpy(),
            'bone_loss': epoch_bone_loss.result().numpy(),
            'motion_loss': epoch_motion_loss.result().numpy(),
            'anatomical_loss': epoch_anatomical_loss.result().numpy(),
            'semantic_loss': epoch_semantic_loss.result().numpy(),
            'diversity_score': diversity_score
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
            best_discriminator_weights = get_model_weights(discriminator.full_model)
        else:
            patience_counter += 1
            logging.info(f"No improvement in loss. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                if best_generator_weights is not None and best_discriminator_weights is not None:
                    logging.info("Restoring best model weights")
                    set_model_weights(generator, best_generator_weights)
                    set_model_weights(discriminator.full_model, best_discriminator_weights)
                break

    # Evaluate final diversity and word variance metrics
    test_noise = tf.random.normal([100, CGAN_NOISE_DIM])  # Generate multiple samples
    test_word_vectors = get_diverse_word_vectors(100)  # Multiple samples across word classes
    
    test_inputs = tf.concat([test_noise, test_word_vectors], axis=1)
    generated_skeletons = generator(test_inputs)
    
    real_skeletons = skeleton_sequences
    pkd_score = calculate_pkd(real_skeletons, generated_skeletons)
    kld_score = calculate_kld(real_skeletons.flatten(), generated_skeletons.flatten())
    diversity_score = calculate_diversity_score(generated_skeletons)
    
    # Check if generated outputs for different words are actually different
    word_output_variance = calculate_word_output_variance(word_specific_outputs)
    
    logging.info(f"Training complete - Total Generator Loss: {total_gen_loss:.4f}, Total Discriminator Loss: {total_disc_loss:.4f}")
    logging.info(f"Final metrics - PKD: {pkd_score:.4f}, KLD: {kld_score:.4f}, Diversity: {diversity_score:.4f}, Word Variance: {word_output_variance:.4f}")
    
    return True, {
        'total_gen_loss': total_gen_loss, 
        'total_disc_loss': total_disc_loss,
        'pkd_score': pkd_score,
        'kld_score': kld_score,
        'diversity_score': diversity_score,
        'word_output_variance': word_output_variance
    }

# Helper function to calculate variance between outputs for different words
def calculate_word_output_variance(word_outputs):
    # For each epoch, calculate the average pairwise distance between outputs for different words
    epoch_variances = []
    
    for epoch in range(len(next(iter(word_outputs.values())))):
        epoch_samples = [word_outputs[word_idx][epoch] for word_idx in word_outputs.keys()]
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(epoch_samples)):
            for j in range(i+1, len(epoch_samples)):
                dist = np.mean(np.square(epoch_samples[i] - epoch_samples[j]))
                distances.append(dist)
        
        epoch_variances.append(np.mean(distances))
    
    return np.mean(epoch_variances)


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
    total_labels = []  # New: store word class labels

    for i in range(0, len(jsonl_files), FILES_PER_BATCH):
        file_batch = jsonl_files[i:i + FILES_PER_BATCH]
        logging.info(f"Processing batch {i//FILES_PER_BATCH + 1}/{-(-len(jsonl_files)//FILES_PER_BATCH)}")
        sequences, vectors, labels = process_file_batch_with_labels(file_batch, word_embeddings)
        if sequences is not None:
            total_sequences.extend(sequences)
            total_vectors.extend(vectors)
            total_labels.extend(labels)

    if not total_sequences:
        logging.error("No valid sequences loaded")
        return

    all_skeleton_sequences = np.array(total_sequences)
    all_word_vectors = np.array(total_vectors)
    all_word_labels = np.array(total_labels)  # Convert labels to numpy array

    # Shuffle the data before training
    all_word_vectors, all_skeleton_sequences, all_word_labels = shuffle_training_data(
        all_word_vectors, all_skeleton_sequences, all_word_labels
    )

    del total_sequences, total_vectors, total_labels, word_embeddings
    gc.collect()

    generator = build_generator()
    discriminator = build_discriminator()

    log_model_summary(generator, "Generator")
    log_model_summary(discriminator, "Discriminator")
    log_training_config()

    try:
        success, history = train_gan(
            generator, 
            discriminator, 
            all_word_vectors, 
            all_skeleton_sequences, 
            all_word_labels,  # Pass the labels
            epochs=CGAN_EPOCHS, 
            batch_size=CGAN_BATCH_SIZE, 
            patience=20
        )
        
        if success:
            save_model_and_history(CGAN_MODEL_PATH, generator, history)
            save_model_and_history(CGAN_DIS_PATH, discriminator)
            
            # Log the new metrics that were returned
            logging.info(f"Per-Keypoint Distance (PKD): {history['pkd_score']:.4f}")
            logging.info(f"KL Divergence (KLD): {history['kld_score']:.4f}")
            logging.info(f"Diversity Score: {history['diversity_score']:.4f}")
            logging.info(f"Word Output Variance: {history['word_output_variance']:.4f}")
        else:
            logging.error("Training failed, models not saved")
    except Exception as e:
        logging.error(f"Training error: {e}")
        return

if __name__ == "__main__":
    setup_logging(f"cgan_training_{MODEL_NAME}")
    main()