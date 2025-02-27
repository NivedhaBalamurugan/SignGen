import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import show_output
from architectures.stgcn import LearnableAdjacency, STGCN
import warnings
from config import *


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, device):
    model = STGCN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def refine_sequence(model, sequence, device):
    with torch.no_grad():
        if sequence.dim() == 3:
            sequence = sequence.unsqueeze(0)
        sequence = sequence.to(device)
        refined = model(sequence)
    return refined.squeeze(0)


def refine_sequence(input_sequence,model_name, isSave_Video):
    model_path = os.path.join(STGCN_MODEL_PATH, "stgcn.pth")
    if not os.path.exists(model_path):
        print("Model file not found. Please ensure the best model is saved at", model_path)
        exit(1)
    model = load_model(model_path, device)
    refined_sequence = refine_sequence(model, input_sequence, device)
    print("Refined sequence shape:", refined_sequence.shape)

    if isSave_Video:
        show_output.save_generated_sequence(refined_sequence, model_name) 

    return refined_sequence

    