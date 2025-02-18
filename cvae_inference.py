import torch
import numpy as np
import os
from cvae import ConditionalVAE  
import show_output

def embedding_for_word(input_word, filepath='Dataset/Glove/glove.6B.50d.txt', embedding_dim=50):

    embedding_matrix = {}
    with open(filepath, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embedding_matrix[word] = vector

    return embedding_matrix.get(input_word, np.zeros(embedding_dim))


SEQ_LEN = 233  
INPUT_DIM = 147
HIDDEN_DIM = 128
LATENT_DIM = 64
COND_DIM = 50  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalVAE(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM,
    cond_dim=COND_DIM,
    output_dim=INPUT_DIM,
    seq_len=SEQ_LEN
).to(device)



checkpoint_path = "cvae_model.pth"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError("Trained model file 'cvae_model.pth' not found.")

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print("Model loaded successfully.")

def generate_sequence(asl_word, model, glove_path, seq_len):
   
    model.eval()
    cond_vector = embedding_for_word(asl_word, filepath=glove_path)
    cond_vector = torch.tensor(cond_vector, dtype=torch.float32).to(device)

    z = torch.randn(1, model.encoder.fc_mu.out_features).to(device)

    with torch.no_grad():
        generated_sequence = model.decoder(z, cond_vector.unsqueeze(0))

    return generated_sequence.squeeze(0).cpu().numpy()

if __name__ == "__main__":
    asl_word = "hello"
    GLOVE_PATH = "Dataset/Glove/glove.6B.50d.txt"
    generated_skeleton = generate_sequence(asl_word, model, GLOVE_PATH, SEQ_LEN)

    print(f"Generated sequence shape: {generated_skeleton.shape}")
    show_output.save_generated_sequence(generated_skeleton, "Dataset/output_sequence") 


