import os
import numpy as np
import tensorflow as tf
from config import *

def save_model_and_history(model_path, model, history=None):
    try:
        model.save(model_path)
        model_name = model_path.split('/')[-1].split('.')[0].split('_')[0] 
        if history is not None:
            history_path = os.path.join(os.path.dirname(model_path), f'{model_name}_training_history.npy')
            np.save(history_path, history)
            logging.info(f"Model saved to {model_path}")
            logging.info(f"Training history saved to {history_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving model or history: {e}")
        return False

def log_model_summary(model, name):
    logging.info(f"{name} Architecture:")
    model.summary(print_fn=logging.info)

def log_training_config():
    logging.info("Training Configuration:")
    logging.info(f"Batch Size: {CGAN_BATCH_SIZE}")
    logging.info(f"Epochs: {CGAN_EPOCHS}")
    logging.info(f"Noise Dimension: {CGAN_NOISE_DIM}")
    logging.info(f"Max Frames: {MAX_FRAMES}")