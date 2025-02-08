import numpy as np

def shear_landmarks(landmarks,shear_x=0.15, shear_y=-0.1):
   
    shear_matrix = np.array([[1, shear_x, 0],  
                             [shear_y, 1, 0],  
                             [0, 0, 1]])  
    landmarks_sheared = landmarks @ shear_matrix.T  
    return landmarks_sheared

def translate_landmarks(landmarks, tx=30, ty=-20, tz=0.0):
    translation_vector = np.array([tx, ty, tz]) 
    return landmarks + translation_vector  

def scale_landmarks(landmarks, scale_x=1.2, scale_y=1.2, scale_z=1.0):
    scaling_matrix = np.array([scale_x, scale_y, scale_z])
    return landmarks * scaling_matrix

def augment_skeleton_sequence(landmarks):
   
    sheared = []
    translated = []
    scaled = []
    
    #for frame in frame_landmarks:

    #landmarks = np.array(frame)
    shearing = shear_landmarks(landmarks)
    translation = translate_landmarks(landmarks)
    scaling = scale_landmarks(landmarks)

    sheared.append(shearing.tolist())
    translated.append(translation.tolist())
    scaled.append(scaling.tolist())       

    augmented = [sheared, translated, scaled ]
    return shearing


