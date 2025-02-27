import os
import glob
import gc
from tqdm import tqdm
import psutil
import numpy as np
import tensorflow as tf
from config import *
from utils.data_utils import load_skeleton_sequences, prepare_training_data
from utils.validation_utils import validate_data_shapes, validate_config
from utils.model_utils import save_model_and_history, log_model_summary, log_training_config
from utils.glove_utils import validate_word_embeddings
from architectures.cgan import build_generator, build_discriminator, discriminator_loss

FILES_PER_BATCH = 2
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

def train_gan(generator, discriminator, word_vectors, skeleton_sequences, epochs=100, batch_size=32):

    if not validate_data_shapes(word_vectors, skeleton_sequences):
        return False
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    generator_optimizer = tf.keras.optimizers.Adam(CGAN_LEARNING_RATE)
    discriminator_optimizer = tf.keras.optimizers.Adam(CGAN_LEARNING_RATE)

    num_samples = word_vectors.shape[0]
    num_batches = max(1, num_samples // batch_size)
    trimmed_size = num_batches * batch_size
    chunks_per_epoch = max(1, trimmed_size // MAX_SAMPLES_PER_BATCH)

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
        
        for chunk_idx in range(chunks_per_epoch):
            chunk_start = chunk_idx * MAX_SAMPLES_PER_BATCH
            chunk_end = min(chunk_start + MAX_SAMPLES_PER_BATCH, trimmed_size)
            
            word_vectors_chunk = word_vectors[chunk_start:chunk_end]
            skeleton_sequences_chunk = skeleton_sequences[chunk_start:chunk_end]
            
            chunk_size = chunk_end - chunk_start
            chunk_batches = max(1, chunk_size // batch_size)
            
            with tqdm(total=chunk_batches, 
                     desc=f"Epoch {epoch+1}/{epochs} Chunk {chunk_idx+1}/{chunks_per_epoch}") as pbar:
                
                for i in range(chunk_batches):
                    if check_memory():
                        logging.info("Cleared memory during training")
                    
                    try:
                        batch_indices = np.random.choice(chunk_size, batch_size, replace=False)
                        word_batch = word_vectors_chunk[batch_indices]
                        skeleton_batch = skeleton_sequences_chunk[batch_indices]

                        gen_loss, disc_loss = train_step(word_batch, skeleton_batch)
                        
                        epoch_gen_loss.update_state(gen_loss)
                        epoch_disc_loss.update_state(disc_loss)
                        
                        pbar.set_postfix({
                            'gen_loss': f'{gen_loss:.4f}',
                            'disc_loss': f'{disc_loss:.4f}'
                        })
                        pbar.update(1)
                        
                        global_batch = epoch * num_batches + chunk_idx * chunk_batches + i
                        if (global_batch + 1) % (num_batches * 5) == 0:
                            try:
                                checkpoint_path = os.path.join(
                                    os.path.dirname(CGAN_GEN_PATH), 
                                    f"checkpoint_e{epoch}_c{chunk_idx}_b{i}.keras"
                                )
                                generator.save(checkpoint_path)
                                logging.info(f"Saved checkpoint: {checkpoint_path}")
                            except Exception as e:
                                logging.error(f"Failed to save checkpoint: {e}")
                        
                    except Exception as e:
                        logging.error(f"Error in training batch: {e}")
                        continue
            
            del word_vectors_chunk, skeleton_sequences_chunk
            gc.collect()
        
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

    word_embeddings = WORD_EMBEDDINGS
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
        success, history = train_gan(generator, discriminator, 
                                   all_word_vectors, all_skeleton_sequences,
                                   epochs=CGAN_EPOCHS, batch_size=CGAN_BATCH_SIZE)

        if success:
            save_model_and_history(CGAN_GEN_PATH, generator, history)
            save_model_and_history(CGAN_DIS_PATH, discriminator)
        else:
            logging.error("Training failed, models not saved")
    except Exception as e:
        logging.error(f"Training error: {e}")
        return

if __name__ == "__main__":
    main()