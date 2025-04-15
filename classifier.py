from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json
from mha import fuse_sequences
import os
import tensorflow as tf
from config import *
from new_cvae_inf import get_cvae_sequence

def load_gloss_labels():
    LOCAL_GLOSS_LABELS_PATH = os.path.join("Dataset", "gloss_labels.json")  

    with open(LOCAL_GLOSS_LABELS_PATH, 'r') as file:
        gloss_labels = json.load(file)

    print(f"Loaded {len(gloss_labels)} gloss labels")
    return gloss_labels

def create_test_set(gloss_labels):
    with open(EXTENDED_WORD_PATH, 'r') as file:
        synonyms_dictionary = json.load(file)

    X_test_raw = []
    y_test_raw = []
    
    for key, synonyms in synonyms_dictionary.items():
        fused_sequence = get_cvae_sequence(key, False)  # Shape: (30, 29, 2)

        if key in gloss_labels:
            label_index = gloss_labels.index(key)
            X_test_raw.append(fused_sequence)
            y_test_raw.append(label_index)
    
    X_test_raw = np.array(X_test_raw)  # (samples, 30, 29, 2)
    
    X_test = X_test_raw.reshape(X_test_raw.shape[0], X_test_raw.shape[1], -1)
    
    y_test = tf.keras.utils.to_categorical(y_test_raw, num_classes=len(gloss_labels))
    
    return X_test, y_test

def load_saved_test_set_from_json():
    json_path = os.path.join("Dataset", "test_set.json")
    with open(json_path, "r") as f:
        test_data = json.load(f)

    X_test = np.array(test_data["X_test"], dtype=np.float32)
    y_test = np.array(test_data["y_test"], dtype=np.float32)
    return X_test, y_test

def main():
    gloss_labels = load_gloss_labels()
    model_save_path = os.path.join(MODELS_PATH, "classifier", "best_classifier.h5")
    loaded_model = load_model(model_save_path)

    print("Loaded model summary:")
    loaded_model.summary()
    X_test, y_test = create_test_set(gloss_labels)
    # X_test, y_test = load_saved_test_set_from_json()
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

    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                            target_names=[gloss_labels[i] for i in range(len(gloss_labels))]))

    
if __name__ == "__main__":
    main()


# def predict_sample(sample_idx):
#     sample = X_test[sample_idx:sample_idx+1]
#     prediction = loaded_model.predict(sample)
#     predicted_class = np.argmax(prediction)
#     true_class = np.argmax(y_test[sample_idx])
    
#     print(f"Sample {sample_idx}:")
#     print(f"Predicted class: {gloss_labels[predicted_class]}")
#     print(f"Actual class: {gloss_labels[true_class]}")
#     print(f"Prediction confidence: {prediction[0][predicted_class]:.4f}")

# for i in range(5):
#     predict_sample(i)
#     print()