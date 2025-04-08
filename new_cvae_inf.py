import torch
import numpy as np
from architectures.cvae import *  # Import your model class
import show_output
from config import *
from utils.data_utils import load_word_embeddings
import os


class SignLanguageGenerator:
    def __init__(self, model_path, input_shape=(30, 29, 2), nhid=64, cond_dim=50, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_shape = input_shape
        self.cond_dim = cond_dim
        
        # Initialize model
        self.model = SignLanguageVAE(input_shape, nhid, cond_dim).to(self.device)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Load your GloVe embeddings (make sure this matches your training setup)
        self.glove_embeddings = load_word_embeddings()  # Your existing function
        
    def generate_from_gloss(self, gloss, num_samples=1):
       
        # Get GloVe embedding for the gloss
        cond_vector = self.glove_embeddings.get(gloss, None)
        if cond_vector is None:
            raise ValueError(f"Gloss '{gloss}' not found in embeddings vocabulary")
            
        # Convert to tensor and repeat for num_samples
        cond_tensor = torch.tensor(cond_vector, dtype=torch.float32).to(self.device)
        cond_tensor = cond_tensor.unsqueeze(0).repeat(num_samples, 1)
        
        # Generate sequences
        with torch.no_grad():
            generated = self.model.generate(cond_tensor)
            return generated.cpu().numpy()  
  

def main(gloss):

    generator = SignLanguageGenerator(
        model_path="Models/cvae_model/model_151.pth",
        input_shape=(30, 29, 2),
        nhid=64,
        cond_dim=50
    )
    
    sequence = generator.generate_from_gloss(gloss)
    print(f"Generated sequence for '{gloss}':", sequence.shape)  
    generated_sign = np.squeeze(sequence, axis=0) 
    show_output.save_generated_sequence(generated_sign, CVAE_OUTPUT_FRAMES, CVAE_OUTPUT_VIDEO)
    # print(generated_sign[10,0:28])
    return generated_sign

    
if __name__ == "__main__":
    main("fine")