import os
import glob
import tensorflow as tf
import numpy as np
from config import *
from utils.data_utils import load_skeleton_sequences, prepare_training_data
from utils.validation_utils import validate_data_shapes, validate_config
from utils.model_utils import save_model_and_history, log_model_summary, log_training_config
from utils.glove_utils import load_word_embeddings, validate_word_embeddings
from architectures.cgan import build_generator, build_discriminator, discriminator_loss

def validate_data_shapes(word_vectors, skeleton_sequences):
    if word_vectors.shape[0] != skeleton_sequences.shape[0]:
        logging.error(f"Mismatch in samples: words={word_vectors.shape[0]}, skeletons={skeleton_sequences.shape[0]}")
        return False
    
    if skeleton_sequences.shape[1:] != (MAX_FRAMES, NUM_JOINTS, NUM_COORDINATES):
        logging.error(f"Invalid skeleton shape: {skeleton_sequences.shape[1:]}, expected: ({MAX_FRAMES}, {NUM_JOINTS}, {NUM_COORDINATES})")
        return False
    
    return True

def train_gan(generator, discriminator, word_vectors, skeleton_sequences, epochs=100, batch_size=32):

    if not validate_data_shapes(word_vectors, skeleton_sequences):
        return False
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    generator_optimizer = tf.keras.optimizers.Adam(CGAN_LEARNING_RATE)
    discriminator_optimizer = tf.keras.optimizers.Adam(CGAN_LEARNING_RATE)

    num_samples = word_vectors.shape[0]
    num_batches = max(1, num_samples // batch_size)
    trimmed_size = num_batches * batch_size

    word_vectors = word_vectors[:trimmed_size]
    skeleton_sequences = skeleton_sequences[:trimmed_size]
    
    logging.info(f"Training with {num_batches} batches per epoch")

    history = {
        'generator_loss': [],
        'discriminator_loss': []
    }

    @tf.function
    def train_step(word_vector, real_skeleton):
        noise = tf.random.normal([batch_size, CGAN_NOISE_DIM], dtype=FP_PRECISION)
        word_vector = tf.cast(word_vector, FP_PRECISION)
        generator_input = tf.concat([word_vector, noise], axis=1)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_skeleton = generator(generator_input, training=True)

            real_skeleton = tf.cast(real_skeleton, FP_PRECISION)
            generated_skeleton = tf.cast(generated_skeleton, FP_PRECISION)

            word_vector_expanded = tf.reshape(word_vector, (batch_size, 1, 1, 50))
            word_vector_expanded = tf.tile(word_vector_expanded, [1, MAX_FRAMES, NUM_JOINTS, 1])

            real_input = tf.concat([real_skeleton, word_vector_expanded], axis=-1)
            fake_input = tf.concat([generated_skeleton, word_vector_expanded], axis=-1)

            real_output = discriminator(real_input, training=True)
            fake_output = discriminator(fake_input, training=True)

            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        return gen_loss, disc_loss

    for epoch in range(epochs):
        epoch_gen_loss = tf.keras.metrics.Mean()
        epoch_disc_loss = tf.keras.metrics.Mean()
        
        for i in range(num_batches):
            try:
                batch_indices = np.random.choice(trimmed_size, batch_size, replace=False)
                word_batch = word_vectors[batch_indices]
                skeleton_batch = skeleton_sequences[batch_indices]

                gen_loss, disc_loss = train_step(word_batch, skeleton_batch)
                
                epoch_gen_loss.update_state(gen_loss)
                epoch_disc_loss.update_state(disc_loss)
                
                if (i + 1) % CGAN_LOG_INTERVAL == 0:
                    logging.info(f"Batch {i+1}/{num_batches}: "
                                 f"Gen Loss = {gen_loss:.4f}, "
                                 f"Disc Loss = {disc_loss:.4f}")
                    
            except Exception as e:
                logging.error(f"Error in training batch: {e}")
                return False
            
            # if epoch_gen_loss.result() > CGAN_MAX_GEN_LOSS or epoch_disc_loss.result() > CGAN_MAX_DISC_LOSS:
            #     logging.error(f"Training stopped due to high losses at epoch {epoch+1}")
            #     return False, history
                            
        history['generator_loss'].append(epoch_gen_loss.result().numpy())
        history['discriminator_loss'].append(epoch_disc_loss.result().numpy())

        logging.info(f"Epoch {epoch+1}/{epochs} - "
                    f"Gen Loss: {epoch_gen_loss.result():.4f}, "
                    f"Disc Loss: {epoch_disc_loss.result():.4f}")

    return True, history

def main():
    if not validate_config():
        return

    os.makedirs(os.path.dirname(CGAN_GEN_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CGAN_DIS_PATH), exist_ok=True)

    word_embeddings = load_word_embeddings(GLOVE_TXT_PATH)
    if not word_embeddings or not validate_word_embeddings(word_embeddings, CGAN_NOISE_DIM):
        return

    jsonl_files = glob.glob(FINAL_JSONL_PATHS)
    if not jsonl_files:
        logging.error(f"No JSONL files found matching pattern: {FINAL_JSONL_PATHS}")
        return

    skeleton_data = load_skeleton_sequences(jsonl_files)
    if not skeleton_data:
        return

    all_skeleton_sequences, all_word_vectors = prepare_training_data(skeleton_data, word_embeddings)
    if all_skeleton_sequences is None:
        return

    generator = build_generator()
    discriminator = build_discriminator()

    log_model_summary(generator, "Generator")
    log_model_summary(discriminator, "Discriminator")
    log_training_config()

    success, history = train_gan(generator, discriminator, 
                               all_word_vectors, all_skeleton_sequences,
                               epochs=CGAN_EPOCHS, batch_size=CGAN_BATCH_SIZE)

    if success:
        save_model_and_history(CGAN_GEN_PATH, generator, history)
        save_model_and_history(CGAN_DIS_PATH, discriminator)
    else:
        logging.error("Training failed, models not saved")

if __name__ == "__main__":
    main()