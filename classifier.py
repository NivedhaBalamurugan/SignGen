from tensorflow.keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import json
from mha import fuse_sequences
import os
import tensorflow as tf
from config import *
from new_cvae_inf import get_cvae_sequence
import pickle

def load_gloss_labels():
    gloss_labels_path = os.path.join("Dataset", "gloss_labels.json")
    with open(gloss_labels_path, 'r') as file:
        gloss_labels = json.load(file)
    return gloss_labels

def load_label_encoder():
    label_encoder_path = os.path.join("Dataset", "label_encoder.pkl")
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder

def create_test_set(gloss_labels, train_from_test=False):
    with open(EXTENDED_WORD_PATH, 'r') as file:
        synonyms_dictionary = json.load(file)

    X_test_raw = []
    y_test_raw = []
    
    for key, synonyms in synonyms_dictionary.items():
        fused_sequence = get_cvae_sequence(key, False)  

        if key in gloss_labels:
            X_test_raw.append(fused_sequence)
            y_test_raw.append(key)
    
    X_test_raw = np.array(X_test_raw)  
    y_test_raw = np.array(y_test_raw)


    if train_from_test:
            json_path = os.path.join("Dataset", "test_set_for_train.json")
            existing_data = {"X_test": [], "y_test": []}
            
            # Try to load existing data
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        existing_data = json.load(f)
                    print(f"Loaded existing data with {len(existing_data['X_test'])} samples")
                except json.JSONDecodeError:
                    print(f"Error loading existing file. Will create new file.")
                    
            # Convert new data to list format for JSON
            new_X_test = X_test_raw.tolist()
            new_y_test = y_test_raw.tolist()
            
            # Append new data to existing data
            existing_data["X_test"].extend(new_X_test)
            existing_data["y_test"].extend(new_y_test)
            
            # Save combined data back to file
            with open(json_path, "w") as f:
                json.dump(existing_data, f)

    return X_test_raw, y_test_raw

def load_saved_test_set_from_json():
    json_path = os.path.join("Dataset", "test_set.json")
    with open(json_path, "r") as f:
        test_data = json.load(f)

    X_test = np.array(test_data["X_test"])
    y_test = np.array(test_data["y_test"])

    indices = np.random.permutation(len(X_test))  
    selected_indices = indices[:100]  
    
    X_test = X_test[selected_indices]
    y_test = y_test[selected_indices]
    
    return X_test, y_test

def main():
    gloss_labels = load_gloss_labels()
    label_encoder = load_label_encoder()
    model_save_path = os.path.join(MODELS_PATH, "classifier", "best_classifier.h5")
    loaded_model = load_model(model_save_path)

    print("Loaded model summary:")
    loaded_model.summary()
    
    all_X_test = []
    all_y_test = []
    
    for _ in range(3):
        X_test_tmp, y_test_tmp = create_test_set(gloss_labels)
        all_X_test.append(X_test_tmp)
        all_y_test.append(y_test_tmp)
    
    X_test_raw = np.concatenate(all_X_test, axis=0)
    y_test_raw = np.concatenate(all_y_test, axis=0)

    X_test_extra, y_test_extra = load_saved_test_set_from_json()

    X_test_raw = X_test_raw.reshape(X_test_raw.shape[0], X_test_raw.shape[1], -1)

    X_test = np.concatenate((X_test_raw, X_test_extra), axis=0)
    y_test_raw = np.concatenate((y_test_raw, y_test_extra), axis=0)

    y_test_encoded = label_encoder.transform(y_test_raw)
    y_test = to_categorical(y_test_encoded)

    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    loss, accuracy = loaded_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred = loaded_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    print("Confusion Matrix:")
    print(conf_matrix)

    # print("Classification Report:")
    # print(classification_report(y_true_classes, y_pred_classes, 
    #                         target_names=[gloss_labels[i] for i in range(len(gloss_labels))]))

    
if __name__ == "__main__":
    main()
