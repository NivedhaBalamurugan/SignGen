import torch
import numpy as np
import os
import logging
import warnings
from architectures.cvae import *
from config import *
import show_output
from utils.data_utils import *


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_sequence(model, condition_embedding, latent_size=64):
    model.eval()
    model.to(device)
    if len(condition_embedding.shape) == 1:
        condition_embedding = condition_embedding.unsqueeze(0)
    condition_embedding = condition_embedding.to(device)
    
    with torch.no_grad():
        z = torch.randn(1, latent_size).to(device)
        # z_conditioned = torch.cat([z, condition_embedding], dim=1)
        # generated_sequence = model.decode(z_conditioned)
        generated_sequence = model.decode(z, condition_embedding)
        return generated_sequence.squeeze(0)

def get_cvae_sequence(asl_word, isSave_Video, CVAE_MODEL_PATH):
    model_path = os.path.join(CVAE_MODEL_PATH, "cvae_epoch_83.pth")
    if not os.path.exists(model_path):
        logging.error("Trained model file not found.")
        return None

    logging.info("Loading trained model...")
    model = CVAE(
        num_frames=MAX_FRAMES,
        num_joints=NUM_JOINTS,
        latent_size=latent_size,
        embedding_dim=EMBEDDING_DIM
    ).to(device)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logging.info("Model loaded successfully.")

    embedding_matrix = load_word_embeddings()
    cond_vector = embedding_matrix.get(asl_word, np.zeros(EMBEDDING_DIM))
    cond_vector = torch.tensor(cond_vector, dtype=torch.float32).to(device)
    
    generated_skeleton = generate_sequence(model, cond_vector, latent_size=latent_size)
    logging.info(f"Generated sequence shape: {generated_skeleton.shape}")

    if isSave_Video:
        show_output.save_generated_sequence(generated_skeleton, CVAE_OUTPUT_FRAMES, CVAE_OUTPUT_VIDEO) 
    
    return generated_skeleton   

if __name__ == "__main__":
    INPUT_WORD = "police"
    get_cvae_sequence(INPUT_WORD, True, CVAE_MODEL_PATH)