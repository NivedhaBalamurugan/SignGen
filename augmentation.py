import numpy as np
from config import FP_PRECISION

DECIMAL_PRECISION = 5
PARAM_PRECISION = 2

def truncate(value, decimals):
    factor = 10 ** decimals
    return np.trunc(value * factor) / factor

def generate_shear_params():
    # Shearing parameters
    shear_x = truncate(np.float32(np.random.uniform(-0.03, 0.03)), PARAM_PRECISION)
    shear_y = truncate(np.float32(np.random.uniform(-0.03, 0.03)), PARAM_PRECISION)
    return shear_x, shear_y

# def generate_translation_params():
#     # Translation parameters
#     tx = int(truncate(np.random.uniform(-3, 3), 0))
#     ty = int(truncate(np.random.uniform(-3, 3), 0))
#     return tx, ty

def generate_scale_params():
    # Scaling parameters
    scale_x = truncate(np.float32(np.random.uniform(0.97, 1.03)), PARAM_PRECISION)
    scale_y = truncate(np.float32(np.random.uniform(0.97, 1.03)), PARAM_PRECISION)
    return scale_x, scale_y

def generate_params():
    shear_x, shear_y = generate_shear_params()
    # tx, ty = generate_translation_params()  # Commented out
    scale_x, scale_y = generate_scale_params()
    
    return (shear_x, shear_y), (0, 0), (scale_x, scale_y)  # Replace translation with (0, 0)

def shear_landmarks(landmarks, shear_params):
    shear_x, shear_y = shear_params
    
    landmarks_xy = landmarks[:, :2]
    shear_matrix = np.array([[1, shear_x], 
                            [shear_y, 1]], dtype=FP_PRECISION)
    
    landmarks_sheared_xy = landmarks_xy @ shear_matrix.T
    landmarks_sheared = landmarks.copy()
    landmarks_sheared[:, :2] = landmarks_sheared_xy
    return truncate(landmarks_sheared, DECIMAL_PRECISION)

# def translate_landmarks(landmarks, trans_params):
#     tx, ty = trans_params
#     
#     landmarks_translated = landmarks.copy()
#     landmarks_translated[:, 0] += tx
#     landmarks_translated[:, 1] += ty
#     return truncate(landmarks_translated, DECIMAL_PRECISION)

def scale_landmarks(landmarks, scale_params):
    scale_x, scale_y = scale_params
    
    landmarks_scaled = landmarks.copy()
    landmarks_scaled[:, 0] *= scale_x
    landmarks_scaled[:, 1] *= scale_y
    return truncate(landmarks_scaled, DECIMAL_PRECISION)

def combine_shear_scale(landmarks, shear_params, scale_params):
    """Apply shearing followed by scaling"""
    landmarks = shear_landmarks(landmarks, shear_params)
    landmarks = scale_landmarks(landmarks, scale_params)
    return truncate(landmarks, DECIMAL_PRECISION)

# def combine_shear_translate(landmarks, shear_params, trans_params):
#     """Apply shearing followed by translation"""
#     landmarks = shear_landmarks(landmarks, shear_params)
#     landmarks = translate_landmarks(landmarks, trans_params)
#     return truncate(landmarks, DECIMAL_PRECISION)

# def combine_scale_translate(landmarks, scale_params, trans_params):
#     """Apply scaling followed by translation"""
#     landmarks = scale_landmarks(landmarks, scale_params)
#     landmarks = translate_landmarks(landmarks, trans_params)
#     return truncate(landmarks, DECIMAL_PRECISION)

def augment_skeleton_sequence_combined(landmarks, shear_params, trans_params, scale_params):
    """Apply augmentations to landmarks using pre-generated parameters"""
    landmarks = np.array(landmarks, dtype=FP_PRECISION)
    
    augmented_versions = {
        # Individual augmentations
        'original': landmarks,
        'shear': shear_landmarks(landmarks, shear_params),
        #'translate': translate_landmarks(landmarks, trans_params),
        'scale': scale_landmarks(landmarks, scale_params),
        
        # Combined augmentations
        'shear_scale': combine_shear_scale(landmarks, shear_params, scale_params),
        # Commented out translation-based combinations
    }
    
    return augmented_versions

def random_combine(landmarks, shear_params, scale_params, trans_params, p=0.5):
    """Randomly combine augmentations with probability p using given parameters"""
    landmarks = np.array(landmarks, dtype=FP_PRECISION)
    
    apply_shear = np.random.random() < p
    apply_scale = np.random.random() < p
    # apply_translate = np.random.random() < p  # Commented out
    
    if apply_shear:
        landmarks = shear_landmarks(landmarks, shear_params)
    if apply_scale:
        landmarks = scale_landmarks(landmarks, scale_params)
    # if apply_translate:
    #     landmarks = translate_landmarks(landmarks, trans_params)
        
    return truncate(landmarks, DECIMAL_PRECISION)