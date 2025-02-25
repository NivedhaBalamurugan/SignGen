import numpy as np

FP_PRECISION = np.float32
DECIMAL_PRECISION = 5
PARAM_PRECISION = 2


def truncate(value, decimals):
    factor = 10 ** decimals
    return np.trunc(value * factor) / factor

def generate_params():
    # Shearing parameters
    shear_x = truncate(np.float32(np.random.uniform(0.1, 0.2)), PARAM_PRECISION)
    shear_y = truncate(np.float32(np.random.uniform(-0.2, -0.1)), PARAM_PRECISION)
    
    # Translation parameters
    tx = int(truncate(np.random.uniform(20, 40), 0))
    ty = int(truncate(np.random.uniform(-30, -10), 0))
    tz = int(truncate(np.random.uniform(-5, 5), 0))
    
    # Scaling parameters
    scale_x = truncate(np.float32(np.random.uniform(1.1, 1.3)), PARAM_PRECISION)
    scale_y = truncate(np.float32(np.random.uniform(1.1, 1.3)), PARAM_PRECISION)
    scale_z = truncate(np.float32(np.random.uniform(0.9, 1.1)), PARAM_PRECISION)
    
    return (shear_x, shear_y), (tx, ty, tz), (scale_x, scale_y, scale_z)

def shear_landmarks(landmarks, shear_params):
    shear_x, shear_y = shear_params
    shear_matrix = np.array([[1, shear_x, 0], 
                            [shear_y, 1, 0], 
                            [0, 0, 1]], dtype=FP_PRECISION)
    
    landmarks_sheared = landmarks @ shear_matrix.T
    return truncate(landmarks_sheared, DECIMAL_PRECISION)

def translate_landmarks(landmarks, trans_params):
    tx, ty, tz = trans_params
    translation_vector = np.array([tx, ty, tz], dtype=FP_PRECISION)
    return truncate(landmarks + translation_vector, DECIMAL_PRECISION)

def scale_landmarks(landmarks, scale_params):
    scale_x, scale_y, scale_z = scale_params
    scaling_matrix = np.array([scale_x, scale_y, scale_z], dtype=FP_PRECISION)
    return truncate(landmarks * scaling_matrix, DECIMAL_PRECISION)

def augment_skeleton_sequence(frame_landmarks):
    landmarks = np.array(frame_landmarks, dtype=FP_PRECISION)
    shear_params, trans_params, scale_params = generate_params()
    
    shearing = shear_landmarks(landmarks, shear_params)
    translation = translate_landmarks(landmarks, trans_params)
    scaling = scale_landmarks(landmarks, scale_params)
    
    return shearing, translation, scaling