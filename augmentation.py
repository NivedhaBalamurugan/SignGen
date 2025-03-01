import numpy as np
from config import FP_PRECISION

DECIMAL_PRECISION = 5
PARAM_PRECISION = 2

def truncate(value, decimals):
    factor = 10 ** decimals
    return np.trunc(value * factor) / factor

def generate_shear_params():
    # Shearing parameters
    shear_x = truncate(np.float32(np.random.uniform(0.1, 0.2)), PARAM_PRECISION)
    shear_y = truncate(np.float32(np.random.uniform(-0.2, -0.1)), PARAM_PRECISION)
    return shear_x, shear_y

def generate_translation_params():
    # Translation parameters
    tx = int(truncate(np.random.uniform(10, 20), 0))
    ty = int(truncate(np.random.uniform(-15, 15), 0))
    return tx, ty

def generate_scale_params():
    # Scaling parameters
    scale_x = truncate(np.float32(np.random.uniform(1.1, 1.3)), PARAM_PRECISION)
    scale_y = truncate(np.float32(np.random.uniform(1.1, 1.3)), PARAM_PRECISION)
    return scale_x, scale_y

def generate_params():
    shear_x, shear_y = generate_shear_params()
    tx, ty = generate_translation_params()
    scale_x, scale_y = generate_scale_params()
    
    return (shear_x, shear_y), (tx, ty), (scale_x, scale_y)

def shear_landmarks(landmarks, shear_params):
    shear_x, shear_y = shear_params
    
    landmarks_xy = landmarks[:, :2]
    shear_matrix = np.array([[1, shear_x], 
                            [shear_y, 1]], dtype=FP_PRECISION)
    
    landmarks_sheared_xy = landmarks_xy @ shear_matrix.T
    landmarks_sheared = landmarks.copy()
    landmarks_sheared[:, :2] = landmarks_sheared_xy
    return truncate(landmarks_sheared, DECIMAL_PRECISION)

def translate_landmarks(landmarks, trans_params):
    tx, ty = trans_params
    
    landmarks_translated = landmarks.copy()
    landmarks_translated[:, 0] += tx
    landmarks_translated[:, 1] += ty
    return truncate(landmarks_translated, DECIMAL_PRECISION)

def scale_landmarks(landmarks, scale_params):
    scale_x, scale_y = scale_params
    
    landmarks_scaled = landmarks.copy()
    landmarks_scaled[:, 0] *= scale_x
    landmarks_scaled[:, 1] *= scale_y
    return truncate(landmarks_scaled, DECIMAL_PRECISION)

def augment_skeleton_sequence(frame_landmarks, shear_params, trans_params, scale_params):
    landmarks = np.array(frame_landmarks, dtype=FP_PRECISION)
    
    shearing = shear_landmarks(landmarks, shear_params)
    translation = translate_landmarks(landmarks, trans_params)
    scaling = scale_landmarks(landmarks, scale_params)
    
    return shearing, translation, scaling