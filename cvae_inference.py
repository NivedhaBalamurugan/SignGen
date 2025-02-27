import torch
import numpy as np
import os
from cvae import ConditionalVAE  
import show_output
from config import *

def generate_sequence(asl_word, model):
   
    model.eval()
    embedding_matrix = WORD_EMBEDDINGS
    cond_vector = embedding_matrix.get(asl_word, np.zeros(embedding_dim))
    cond_vector = torch.tensor(cond_vector, dtype=torch.float32).to(device)

    z = torch.randn(1, model.encoder.fc_mu.out_features).to(device)

    with torch.no_grad():
        generated_sequence = model.decoder(z, cond_vector.unsqueeze(0))

    return generated_sequence.squeeze(0).cpu().numpy()


def get_cvae_sequence(asl_word, isSave_Video):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalVAE(
        input_dim=CVAE_INPUT_DIM,
        hidden_dim=CVAE_HIDDEN_DIM,
        latent_dim=CVAE_LATENT_DIM,
        cond_dim=EMBEDDING_DIM,
        output_dim=CVAE_INPUT_DIM,
        seq_len=MAX_FRAMES
    ).to(device)


    if not os.path.exists(CVAE_MODEL_PATH):
        logging.error("Trained model file  not found.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logging.info("Model loaded successfully.")

    generated_skeleton = generate_sequence(asl_word, model)

    logging.info(f"Generated sequence shape: {generated_skeleton.shape}")
    if isSave_Video:
        show_output.save_generated_sequence(generated_skeleton) 
    return generated_skeleton