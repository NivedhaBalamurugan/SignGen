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
        fused_sequence,ssim = get_cvae_sequence(key, False)  

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

# def main():
#     gloss_labels = load_gloss_labels()
#     label_encoder = load_label_encoder()
#     model_save_path = os.path.join(MODELS_PATH, "classifier", "best_classifier.h5")
#     loaded_model = load_model(model_save_path)

#     print("Loaded model summary:")
#     loaded_model.summary()
    
#     all_X_test = []
#     all_y_test = []
    
#     for _ in range(3):
#         X_test_tmp, y_test_tmp = create_test_set(gloss_labels)
#         all_X_test.append(X_test_tmp)
#         all_y_test.append(y_test_tmp)
    
#     X_test_raw = np.concatenate(all_X_test, axis=0)
#     y_test_raw = np.concatenate(all_y_test, axis=0)

#     X_test_extra, y_test_extra = load_saved_test_set_from_json()

#     X_test_raw = X_test_raw.reshape(X_test_raw.shape[0], X_test_raw.shape[1], -1)

#     X_test = np.concatenate((X_test_raw, X_test_extra), axis=0)
#     y_test_raw = np.concatenate((y_test_raw, y_test_extra), axis=0)

#     y_test_encoded = label_encoder.transform(y_test_raw)
#     y_test = to_categorical(y_test_encoded)

#     print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

#     loss, accuracy = loaded_model.evaluate(X_test, y_test)
#     print(f"Test Loss: {loss:.4f}")
#     print(f"Test Accuracy: {accuracy:.4f}")

#     y_pred = loaded_model.predict(X_test)
#     y_pred_classes = np.argmax(y_pred, axis=1)
#     y_true_classes = np.argmax(y_test, axis=1)

#     conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
#     print("Confusion Matrix:")
#     print(conf_matrix)

#     # print("Classification Report:")
#     # print(classification_report(y_true_classes, y_pred_classes, 
#     #                         target_names=[gloss_labels[i] for i in range(len(gloss_labels))]))


def test_single_word(word, model, label_encoder, gloss_labels):
   
    if word not in gloss_labels:
        print(f"Word '{word}' not found in gloss labels")
        return
    
    # Get the sequence for this word
    fused_sequence, ssim = get_cvae_sequence(word, False)
    
    if fused_sequence is None:
        print(f"Could not generate sequence for word '{word}'")
        return
    
    # Prepare the input (add batch dimension)
    X_test = np.array([fused_sequence])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
    
    # Get the true label
    y_true = word
    y_true_encoded = label_encoder.transform([y_true])
    y_true_onehot = to_categorical(y_true_encoded, num_classes=len(gloss_labels))
    
    # Make prediction
    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_pred_word = label_encoder.inverse_transform(y_pred_class)[0]
    
    # Get top 3 predictions
    top3_indices = np.argsort(y_pred[0])[-3:][::-1]
    top3_words = label_encoder.inverse_transform(top3_indices)
    top3_probs = y_pred[0][top3_indices]
    
    # Print results
    print(f"\nTesting word: {word}")
    print(f"Generated sequence shape: {fused_sequence.shape}")
    print(f"True label: {y_true}")
    print(f"Predicted label: {y_pred_word}")
    print(f"Prediction confidence: {y_pred[0][y_pred_class][0]:.2f}")
    # print("\nTop 3 predictions:")
    # for word, prob in zip(top3_words, top3_probs):
    #     print(f"  {word}: {prob:.4f}")
    
    # Check if correct
    if y_true == y_pred_word:
        print("\n✅ Correct prediction!")
    else:
        print("\n❌ Incorrect prediction")

def main():
    gloss_labels = load_gloss_labels()
    label_encoder = load_label_encoder()
    model_save_path = os.path.join(MODELS_PATH, "classifier", "best_classifier.h5")
    loaded_model = load_model(model_save_path)

    print("Loaded model summary:")
    loaded_model.summary()
    
    # Test a specific word
    test_word = "computer"  # Change this to the word you want to test
    test_single_word(test_word, loaded_model, label_encoder, gloss_labels)
    
    # You could also test multiple words in a loop:
    # test_words = ["apple", "banana", "hello", "goodbye"]
    # for word in test_words:
    #     test_single_word(word, loaded_model, label_encoder, gloss_labels)


if __name__ == "__main__":
    main()
