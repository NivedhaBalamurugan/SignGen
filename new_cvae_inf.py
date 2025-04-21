import torch
import numpy as np
from architectures.cvae import *  
import show_output
from config import *
from utils.data_utils import load_word_embeddings, get_real_data
from skimage.metrics import structural_similarity as ssim

class SignLanguageGenerator:
    def __init__(self, model_path, input_shape=(30, 29, 2), nhid=64, cond_dim=50, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_shape = input_shape
        self.cond_dim = cond_dim
        
        self.model = SignLanguageVAE(input_shape, nhid, cond_dim).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        self.glove_embeddings = load_word_embeddings()  
        
    def generate_from_gloss(self, gloss, num_samples=1):
        cond_vector = self.glove_embeddings.get(gloss, None)
        if cond_vector is None:
            raise ValueError(f"Gloss '{gloss}' not found in embeddings vocabulary")
            
        cond_tensor = torch.tensor(cond_vector, dtype=torch.float32).to(self.device)
        cond_tensor = cond_tensor.unsqueeze(0).repeat(num_samples, 1)
        
        with torch.no_grad():
            generated = self.model.generate(cond_tensor)
            return generated.cpu().numpy()  


def compute_ssim_score(generated_seq, gloss):

    ground_truth_seq = get_real_data(gloss)  
    ground_truth_seq = np.array(ground_truth_seq)

    if ground_truth_seq is None:
        print(f"No ground truth sequence found for '{gloss}'")
        return None
    else:
        if generated_seq.shape != ground_truth_seq.shape:
            print(generated_seq.shape, ground_truth_seq.shape)
            raise ValueError("Generated and ground truth sequences must have the same shape")
        
        ssim_scores = []
        num_frames = generated_seq.shape[0]
        
        for i in range(num_frames):
            gen_frame = generated_seq[i]  
            gt_frame = ground_truth_seq[i]          

            gen_frame_flat = gen_frame.flatten()
            gt_frame_flat = gt_frame.flatten()
            
            data_range = max(gen_frame_flat.max() - gen_frame_flat.min(), gt_frame_flat.max() - gt_frame_flat.min())
            if data_range == 0:
                data_range = 1.0  
            
            score = ssim(gen_frame_flat, gt_frame_flat, data_range=data_range)
            ssim_scores.append(score)
    
    ssim_score = np.mean(ssim_scores)

    return ssim_score 

def get_cvae_sequence(word, isSave=True):
    gloss = check_extended_words(word.lower())
    generator = SignLanguageGenerator(
        model_path="Models/cvae_model/model_151.pth",
        input_shape=(30, 29, 2),
        nhid=64,
        cond_dim=50
    )
    
    sequence = generator.generate_from_gloss(gloss)
    print(f"Generated CVAE sequence for '{word}':", sequence.shape)  
    generated_sign = np.squeeze(sequence, axis=0)  
    
    ssim_score = compute_ssim_score(generated_sign, gloss)
    print(f"SSIM score for '{word}': {ssim_score}")   
        
    
    if isSave:
        show_output.save_generated_sequence(generated_sign, CVAE_OUTPUT_FRAMES, CVAE_OUTPUT_VIDEO, "cvae")

    return generated_sign, ssim_score


if __name__ == "__main__":
    generated_sign, ssim_score = get_cvae_sequence("movie")
