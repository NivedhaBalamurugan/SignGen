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


import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchsummary import summary
from torchinfo import summary


def compute_mse(original, reconstructed):
    return torch.mean((original - reconstructed) ** 2).item()

def compute_mae(original, reconstructed):
    return torch.mean(torch.abs(original - reconstructed)).item()

def compute_psnr(mse, max_val=1.0):
    return 10 * torch.log10(max_val ** 2 / mse).item()

def compute_ssim(original, reconstructed):
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    ssim_val = 0
    for i in range(original.shape[0]):  # Loop over batch
        ssim_val += ssim(original[i], reconstructed[i], multichannel=True)
    return ssim_val / original.shape[0]

def evaluate_model(model, dataloader, device):

    logging.info("Evaluating metris")
    model.eval()
    mse_values, ssim_values = [], []
    latent_vectors = []
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            condition = batch['condition'].to(device)
            recon_video, mu, logvar, _, z = model(video, condition)
            
            # Compute metrics
            mse = compute_mse(video, recon_video)
            ssim_val = compute_ssim(video, recon_video)
            mse_values.append(mse)
            ssim_values.append(ssim_val)
            
            # Collect latent vectors
            latent_vectors.append(z)
    
    # Aggregate results
    avg_mse = np.mean(mse_values)
    avg_ssim = np.mean(ssim_values)
    latent_vectors = torch.cat(latent_vectors, dim=0)
    
    logging.info(f"Average MSE: {avg_mse:.8f}")
    logging.info(f"Average SSIM: {avg_ssim:.8f}")
    


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
    
    # input_shape = [(CVAE_BATCH_SIZE, MAX_FRAMES, CVAE_INPUT_DIM, 1), (CAVE_BATCH_SIZE, EMBEDDING_DIM)]    
    # summary(model, input_size=input_shape, depth=4, col_names=["input_size", "output_size", "num_params", "trainable"])
   # generated_skeleton = generate_sequence(asl_word, model)
   # logging.info(f"Generated sequence shape: {generated_skeleton.shape}")
    
    #if isSave_Video:
     #   show_output.save_generated_sequence(generated_skeleton, CVAE_OUTPUT_FRAMES, CVAE_OUTPUT_VIDEO) 
    
    return generated_skeleton

if __name__ == "__main__":
    get_cvae_sequence("anatomy",1)