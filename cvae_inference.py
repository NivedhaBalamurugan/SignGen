import torch
import numpy as np
import os
import logging
import warnings
from architectures.cvae import ConditionalVAE
from config import *
import show_output

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_sequence(asl_word, model):
    model.eval()
    embedding_matrix = WORD_EMBEDDINGS
    cond_vector = embedding_matrix.get(asl_word, np.zeros(EMBEDDING_DIM))
    cond_vector = torch.tensor(cond_vector, dtype=torch.float32).to(device)
    
    z = torch.randn(1, CVAE_LATENT_DIM).to(device)
    
    with torch.no_grad():
        generated_sequence = model.decoder(z, cond_vector.unsqueeze(0))
    
    return generated_sequence.squeeze(0).cpu().numpy()

def get_cvae_sequence(asl_word, isSave_Video=False):
    model_path = os.path.join(CVAE_MODEL_PATH, "cvae.pth")
    if not os.path.exists(model_path):
        logging.error("Trained model file not found.")
        return None

    logging.info("Loading trained model...")
    model = ConditionalVAE(
        input_dim=CVAE_INPUT_DIM,
        hidden_dim=CVAE_HIDDEN_DIM,
        latent_dim=CVAE_LATENT_DIM,
        cond_dim=EMBEDDING_DIM,
        output_dim=CVAE_INPUT_DIM,
        seq_len=MAX_FRAMES
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    logging.info("Model loaded successfully.")
    
    generated_skeleton = generate_sequence(asl_word, model)
    logging.info(f"Generated sequence shape: {generated_skeleton.shape}")
    
    if isSave_Video:
        show_output.save_generated_sequence(generated_skeleton, CVAE_OUTPUT_FRAMES, CVAE_OUTPUT_VIDEO) 
    
    return generated_skeleton

if __name__ == "__main__":
    get_cvae_sequence("anatomy",1)