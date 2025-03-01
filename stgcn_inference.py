import torch
import numpy as np
from architectures.stgcn import STGCN, create_edge_index, group_joints


def load_model():
    model = STGCN(num_nodes=16, num_features=3, num_classes=3)
    model_save_path = os.path.join(STGCN_MODEL_PATH, "stgcn.pth")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    return model

def refine_generated_sequence(model, generated_sequence, edge_index):
    with torch.no_grad():
        grouped_sequence = group_joints(generated_sequence)
        grouped_sequence = torch.tensor(grouped_sequence, dtype=torch.float32).unsqueeze(0)
        refined_sequence = model(grouped_sequence, edge_index)
        return refined_sequence.squeeze(0).numpy()

def get_refined_sequence(input_sequence, model_name, isSave_Video):
    model = load_model()
    
    edge_index = create_edge_index()

    refined_sequence = refine_generated_sequence(model, input_sequence, edge_index)
    print("Refined sequence shape:", refined_sequence.shape)

    if isSave_Video:
            model_name = model_name + "_STGCN"
            show_output.save_generated_sequence(refined_sequence, model_name) 
     return refined_sequence
