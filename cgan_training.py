import os
import glob
import gc
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

MODEL_NAME = "resume_new_attempt"

def load_all_data(jsonl_gz_files, word_embeddings):
    logging.info(f"Loading data from {len(jsonl_gz_files)} files")
    total_sequences = []
    total_vectors = []

    with tqdm(total=len(jsonl_gz_files), desc="Loading files") as pbar:
        for file_path in jsonl_gz_files:
            try:
                logging.info(f"Processing file: {os.path.basename(file_path)}")
                skeleton_data = load_skeleton_sequences([file_path])
                if skeleton_data:
                    sequences, vectors = prepare_training_data(skeleton_data, word_embeddings)
                    total_sequences.extend(sequences)
                    total_vectors.extend(vectors)
                del skeleton_data
                gc.collect()
                pbar.update(1)
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                pbar.update(1)
                continue

    return np.array(total_sequences), np.array(total_vectors)

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
    checkpoint_dir = os.path.join(os.path.dirname(CGAN_MODEL_PATH), f"checkpoints_{MODEL_NAME}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    gen_weights_path = os.path.join(checkpoint_dir, f"generator_weights_epoch{epoch}_loss{loss:.4f}.weights.weights.h5")
    generator.save_weights(gen_weights_path)

    disc_weights_path = os.path.join(checkpoint_dir, f"discriminator_weights_epoch{epoch}_loss{loss:.4f}.weights.weights.h5")
    discriminator.save_weights(disc_weights_path)

    try:
        gen_model_path = os.path.join(checkpoint_dir, f"generator_epoch{epoch}_loss{loss:.4f}.keras")
        os.makedirs(gen_model_path, exist_ok=True)
        tf.saved_model.save(generator, gen_model_path)

        # disc_model_path = os.path.join(checkpoint_dir, f"discriminator_epoch{epoch}_loss{loss:.4f}")
        # os.makedirs(disc_model_path, exist_ok=True)
        # tf.saved_model.save(discriminator, disc_model_path)

        logging.info(f"Successfully saved full models for epoch {epoch}")
    except Exception as e:
        logging.warning(f"Could not save full models: {e}. Only weights were saved.")

    history_path = os.path.join(checkpoint_dir, f"history_epoch{epoch}.npy")
    np.save(history_path, history)

    logging.info(f"Saved checkpoint for epoch {epoch} with loss {loss:.4f}")
    return gen_weights_path, disc_weights_path

def train_gan(generator, discriminator, word_vectors, skeleton_sequences,
              epochs=100, batch_size=CGAN_BATCH_SIZE, patience=10,
              starting_epoch=0, previous_history=None):
    if not validate_data_shapes(word_vectors, skeleton_sequences):
        return False

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

    best_loss = float('inf')

    if previous_history is not None:
        total_gen_loss = previous_history.get('total_gen_loss', 0.0)
        total_disc_loss = previous_history.get('total_disc_loss', 0.0)
        best_loss = previous_history.get('current_loss', float('inf'))
    else:
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        best_loss = float('inf')

    num_samples = word_vectors.shape[0]
    steps_per_epoch = num_samples // batch_size + (1 if num_samples % batch_size > 0 else 0)

    logging.info(f"Training with {num_samples} samples, {steps_per_epoch} steps per epoch")

    # Create TensorFlow dataset with shuffling
    train_dataset = tf.data.Dataset.from_tensor_slices((word_vectors, skeleton_sequences))

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
            bone_loss = bone_length_consistency_loss(generated_skeleton)
            motion_loss = motion_smoothness_loss(generated_skeleton)
            anatomical_loss = anatomical_plausibility_loss(generated_skeleton)

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

    for epoch in range(starting_epoch, epochs):
        # Shuffle data for each epoch - critical for preventing mode collapse
        shuffled_dataset = train_dataset.shuffle(buffer_size=min(50000, num_samples), reshuffle_each_iteration=True)
        batched_dataset = shuffled_dataset.batch(batch_size, drop_remainder=False)

        epoch_gen_loss = tf.keras.metrics.Mean()
        epoch_disc_loss = tf.keras.metrics.Mean()
        epoch_adv_loss = tf.keras.metrics.Mean()
        epoch_mse_loss = tf.keras.metrics.Mean()
        epoch_bone_loss = tf.keras.metrics.Mean()
        epoch_motion_loss = tf.keras.metrics.Mean()
        epoch_anatomical_loss = tf.keras.metrics.Mean()
        epoch_semantic_loss = tf.keras.metrics.Mean()

        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for word_batch, skeleton_batch in batched_dataset:
                try:
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
                    pbar.update(1)
                    continue

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

def resume_training(checkpoint_epoch, model_name=MODEL_NAME):
    checkpoint_dir = os.path.join(os.path.dirname(CGAN_MODEL_PATH), f"checkpoints_{model_name}")

    gen_weights_files = glob.glob(os.path.join(checkpoint_dir, f"generator_weights_epoch{checkpoint_epoch}_*.weights.h5"))
    disc_weights_files = glob.glob(os.path.join(checkpoint_dir, f"discriminator_weights_epoch{checkpoint_epoch}_*.weights.h5"))
    history_files = glob.glob(os.path.join(checkpoint_dir, f"history_epoch{checkpoint_epoch}.npy"))

    if not gen_weights_files or not disc_weights_files or not history_files:
        logging.error(f"Could not find checkpoint files for epoch {checkpoint_epoch}")
        return None, None, None

    generator = build_generator()
    discriminator = build_discriminator()

    try:
        generator.load_weights(gen_weights_files[0])
        logging.info(f"Successfully loaded generator weights from {gen_weights_files[0]}")

        discriminator.load_weights(disc_weights_files[0])
        logging.info(f"Successfully loaded discriminator weights from {disc_weights_files[0]}")

        history = np.load(history_files[0], allow_pickle=True).item()
        logging.info(f"Resumed from epoch {checkpoint_epoch} with loss {history.get('current_loss', 'N/A')}")

        return generator, discriminator, history
    except Exception as e:
        logging.error(f"Error loading weights: {e}")
        return None, None, None

def main(resume_epoch=None):
    logging.info("Starting CGAN training process...")
    if not validate_config():
        return

    os.makedirs(os.path.dirname(CGAN_MODEL_PATH), exist_ok=True)
    #os.makedirs(os.path.dirname(CGAN_DIS_PATH), exist_ok=True)

    word_embeddings = load_word_embeddings()
    if not word_embeddings or not validate_word_embeddings(word_embeddings):
        return

    jsonl_gz_files = sorted(glob.glob(FINAL_JSONL_GZ_PATHS))
    if not jsonl_gz_files:
        logging.error(f"No JSONL files found matching pattern: {FINAL_JSONL_GZ_PATHS}")
        return

    all_skeleton_sequences, all_word_vectors = load_all_data(jsonl_gz_files, word_embeddings)

    if len(all_skeleton_sequences) == 0:
        logging.error("No valid sequences loaded")
        return

    logging.info(f"Loaded {len(all_skeleton_sequences)} sequences for training")

    del word_embeddings
    gc.collect()

    if resume_epoch:
        generator, discriminator, history = resume_training(resume_epoch)
        if generator is None:
            logging.error("Failed to resume training. Starting from scratch.")
            generator = build_generator()
            discriminator = build_discriminator()
            starting_epoch = 0
            previous_history = None
        else:
            starting_epoch = resume_epoch
            previous_history = history
    else:
        generator = build_generator()
        discriminator = build_discriminator()
        starting_epoch = 0
        previous_history = None

    log_model_summary(generator, "Generator")
    log_model_summary(discriminator, "Discriminator")
    log_training_config()

    try:
        success, history = train_gan(
            generator,
            discriminator,
            all_word_vectors,
            all_skeleton_sequences,
            epochs=CGAN_EPOCHS,
            batch_size=CGAN_BATCH_SIZE,
            patience=20,
            starting_epoch=starting_epoch,
            previous_history=previous_history
        )
        if success:
            save_model_and_history(CGAN_MODEL_PATH, generator, history)
            #save_model_and_history(CGAN_DIS_PATH, discriminator)

            num_eval_samples = min(1000, len(all_word_vectors))
            eval_indices = np.random.choice(len(all_word_vectors), num_eval_samples, replace=False)
            eval_vectors = all_word_vectors[eval_indices]
            eval_real = all_skeleton_sequences[eval_indices]

            generated_skeletons = generator(eval_vectors)
            # pkd_score = calculate_pkd(eval_real, generated_skeletons)
            # kld_score = calculate_kld(eval_real.flatten(), generated_skeletons.flatten())

            diversity_score = calculate_diversity_score(generated_skeletons)
            # logging.info(f"Per-Keypoint Distance (PKD): {pkd_score:.4f}")
            # logging.info(f"KL Divergence (KLD): {kld_score:.4f}")
            logging.info(f"Diversity Score: {diversity_score:.4f}")
        else:
            logging.error("Training failed, models not saved")
    except Exception as e:
        logging.error(f"Training error: {e}")
        return

if __name__ == "__main__":
    setup_logging(f"cgan_training_{MODEL_NAME}")
    resume_epoch = 16
    main(resume_epoch=resume_epoch)
    #main()