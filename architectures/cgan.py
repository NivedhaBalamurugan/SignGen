import tensorflow as tf
from keras.layers import GRU, Dense, Input, Reshape, TimeDistributed, Bidirectional, UpSampling1D
from keras.models import Model
from keras.layers import GRU, Dense, Input, Dropout, BatchNormalization, Reshape, Conv1D, LeakyReLU
from config import *

def build_generator(num_segments):
    inputs = Input(shape=(CGAN_NOISE_DIM + EMBEDDING_DIM,))
    
    x = Dense(256, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Reshape((16, 32))(x)
    
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    
    x = TimeDistributed(Dense(128, activation="relu"))(x)
    x = UpSampling1D(2)(x)
    
    x = TimeDistributed(Dense(num_segments * 4, activation="tanh"))(x)
    
    x = x[:, :MAX_FRAMES]
    outputs = Reshape((MAX_FRAMES, num_segments, 2, NUM_COORDINATES))(x)
    
    return Model(inputs, outputs)

def build_discriminator(num_segments):
    # Skeleton input
    skeleton_input = Input(shape=(MAX_FRAMES, num_segments, 2, NUM_COORDINATES))
    # Word embedding input
    embedding_input = Input(shape=(EMBEDDING_DIM,))
    
    # Process skeleton input
    reshape_size = num_segments * 2 * NUM_COORDINATES
    x = Reshape((MAX_FRAMES, reshape_size))(skeleton_input)
    
    x = Conv1D(128, kernel_size=5, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv1D(256, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(256, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = Bidirectional(GRU(128, return_sequences=False))(x)
    
    # Expand and tile the embedding to combine with the skeleton features
    embedding_expanded = Dense(256, activation="relu")(embedding_input)
    
    # Concatenate skeleton features with word embedding
    combined = tf.concat([x, embedding_expanded], axis=-1)
    
    x = Dense(256, activation="relu")(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    
    return Model([skeleton_input, embedding_input], outputs)
def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def discriminator_loss(real_output, fake_output):
    # Wasserstein loss for discriminator
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    return real_loss + fake_loss

def bone_length_consistency_loss(skeletons, joint_connections):
    """
    Penalize variations in bone length across frames
    joint_connections: list of tuples (joint_idx1, joint_idx2) defining bones
    """
    losses = []
    for i in range(len(joint_connections)):
        joint1_idx, joint2_idx = joint_connections[i]
        
        # Get positions of connected joints
        joint1 = skeletons[:, :, joint1_idx, :]  # [batch, frames, coordinates]
        joint2 = skeletons[:, :, joint2_idx, :]  # [batch, frames, coordinates]
        
        # Calculate bone lengths for each frame
        bone_lengths = tf.sqrt(tf.reduce_sum(tf.square(joint1 - joint2), axis=-1) + 1e-8)  # [batch, frames]
        
        # Calculate variance of bone length across frames (should be near 0)
        mean_lengths = tf.reduce_mean(bone_lengths, axis=1, keepdims=True)  # [batch, 1]
        length_variance = tf.reduce_mean(tf.square(bone_lengths - mean_lengths))
        
        losses.append(length_variance)
    
    return tf.reduce_mean(losses)

def motion_smoothness_loss(skeletons):
    """
    Penalize jerky, non-smooth motion between frames
    """
    # Calculate velocity (first derivative of position)
    velocity = skeletons[:, 1:] - skeletons[:, :-1]  # [batch, frames-1, joints, coords]
    
    # Calculate acceleration (second derivative of position)
    acceleration = velocity[:, 1:] - velocity[:, :-1]  # [batch, frames-2, joints, coords]
    
    # Penalize large accelerations (encourage smooth motion)
    return tf.reduce_mean(tf.square(acceleration))

def anatomical_plausibility_loss(skeletons, joint_angle_limits):
    """
    Penalizes anatomically implausible joint angles
    """
    losses = []
    
    for (joint1, joint2, joint3, min_angle, max_angle) in joint_angle_limits:
        # Get joint positions
        pos1 = skeletons[:, :, joint1, :]  # [batch, frames, coordinates]
        pos2 = skeletons[:, :, joint2, :]  # [batch, frames, coordinates]
        pos3 = skeletons[:, :, joint3, :]  # [batch, frames, coordinates]
        
        # Calculate vectors
        v1 = pos1 - pos2  # Vector from joint2 to joint1
        v2 = pos3 - pos2  # Vector from joint2 to joint3
        
        # Calculate dot product
        dot_product = tf.reduce_sum(v1 * v2, axis=-1)
        
        # Calculate magnitudes
        v1_mag = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=-1) + 1e-8)
        v2_mag = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=-1) + 1e-8)
        
        # Calculate cosine of angle
        cos_angle = dot_product / (v1_mag * v2_mag)
        cos_angle = tf.clip_by_value(cos_angle, -0.9999, 0.9999)  # Prevent numerical issues
        
        # Convert to angle in radians
        angle = tf.acos(cos_angle)
        
        # Penalize angles outside the defined limits
        min_rad = min_angle * np.pi / 180
        max_rad = max_angle * np.pi / 180
        
        penalty = tf.maximum(0.0, min_rad - angle) + tf.maximum(0.0, angle - max_rad)
        losses.append(tf.reduce_mean(penalty))
    
    return tf.reduce_mean(losses)

def semantic_consistency_loss(generated_skeletons, word_vectors):
    """
    Ensures that similar words produce similar motions
    """
    # Calculate pairwise cosine similarity between word vectors
    word_similarities = tf.matmul(
        tf.nn.l2_normalize(word_vectors, axis=1),
        tf.nn.l2_normalize(word_vectors, axis=1),
        transpose_b=True
    )
    
    # Calculate motion similarity (use average joint positions)
    flattened_skeletons = tf.reshape(generated_skeletons, 
                                    (tf.shape(generated_skeletons)[0], -1))
    skeleton_similarities = tf.matmul(
        tf.nn.l2_normalize(flattened_skeletons, axis=1),
        tf.nn.l2_normalize(flattened_skeletons, axis=1),
        transpose_b=True
    )
    
    # Loss is higher when word similarity doesn't match motion similarity
    return tf.reduce_mean(tf.square(word_similarities - skeleton_similarities))

def bone_length_consistency_loss_segments(skeletons):
    # Calculate the length of each bone in each frame
    # For each line segment, we calculate the distance between its start and end points
    start_points = skeletons[:, :, :, 0, :]  # Shape: (batch_size, frames, num_segments, 2)
    end_points = skeletons[:, :, :, 1, :]    # Shape: (batch_size, frames, num_segments, 2)
    
    # Calculate the squared distance between start and end points
    bone_lengths = tf.sqrt(tf.reduce_sum(tf.square(end_points - start_points), axis=3) + 1e-8)
    # Shape: (batch_size, frames, num_segments)
    
    # For each bone, calculate the variance of its length across frames
    mean_lengths = tf.reduce_mean(bone_lengths, axis=1, keepdims=True)  # Mean across frames
    length_variance = tf.reduce_mean(tf.square(bone_lengths - mean_lengths), axis=[1, 2])
    
    return tf.reduce_mean(length_variance)

def motion_smoothness_loss_segments(skeletons):
    # Flatten the last two dimensions to simplify calculations
    # New shape: (batch_size, frames, num_segments * 4)
    flat_skeletons = tf.reshape(skeletons, (tf.shape(skeletons)[0], tf.shape(skeletons)[1], -1))
    
    # Calculate velocity and acceleration
    velocity = flat_skeletons[:, 1:] - flat_skeletons[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    
    return tf.reduce_mean(tf.square(acceleration))

def anatomical_plausibility_loss_segments(skeletons, joint_connections):
    # First, we need to create a mapping from segment indices to joint indices
    segment_to_joint = {}
    for i, (j1, j2) in enumerate(joint_connections):
        if j1 not in segment_to_joint:
            segment_to_joint[j1] = []
        if j2 not in segment_to_joint:
            segment_to_joint[j2] = []
        segment_to_joint[j1].append((i, 0))  # (segment_idx, point_idx)
        segment_to_joint[j2].append((i, 1))  # (segment_idx, point_idx)
    
    # Define joint angle constraints
    joint_angle_constraints = [
        # (joint1_idx, center_joint_idx, joint2_idx, min_angle, max_angle)
        (2, 3, 4, 0, 160),   # Right elbow
        (5, 6, 7, 0, 160),   # Left elbow
        # Add more constraints as needed
    ]
    
    losses = []
    batch_size = tf.shape(skeletons)[0]
    num_frames = tf.shape(skeletons)[1]
    
    for joint1_idx, center_idx, joint2_idx, min_angle, max_angle in joint_angle_constraints:
        # Find the segments that contain these joints
        joint1_segments = segment_to_joint.get(joint1_idx, [])
        center_segments = segment_to_joint.get(center_idx, [])
        joint2_segments = segment_to_joint.get(joint2_idx, [])
        
        if not joint1_segments or not center_segments or not joint2_segments:
            continue
        
        # Use the first segment that contains each joint
        j1_seg_idx, j1_point_idx = joint1_segments[0]
        c_seg_idx, c_point_idx = center_segments[0]
        j2_seg_idx, j2_point_idx = joint2_segments[0]
        
        # Extract the joint positions
        joint1_pos = skeletons[:, :, j1_seg_idx, j1_point_idx, :]
        center_pos = skeletons[:, :, c_seg_idx, c_point_idx, :]
        joint2_pos = skeletons[:, :, j2_seg_idx, j2_point_idx, :]
        
        # Calculate vectors
        v1 = joint1_pos - center_pos
        v2 = joint2_pos - center_pos
        
        # Calculate dot product and magnitudes
        dot_product = tf.reduce_sum(v1 * v2, axis=2)
        v1_mag = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=2) + 1e-8)
        v2_mag = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=2) + 1e-8)
        
        # Calculate cosine of angle and then the angle itself
        cos_angle = dot_product / (v1_mag * v2_mag)
        cos_angle = tf.clip_by_value(cos_angle, -0.9999, 0.9999)
        angle = tf.acos(cos_angle)
        
        # Convert angle constraints to radians
        min_rad = min_angle * np.pi / 180
        max_rad = max_angle * np.pi / 180
        
        # Calculate penalty for angles outside the allowed range
        penalty = tf.maximum(0.0, min_rad - angle) + tf.maximum(0.0, angle - max_rad)
        losses.append(tf.reduce_mean(penalty))
    
    if not losses:
        return tf.constant(0.0, dtype=tf.float32)
    
    return tf.reduce_mean(losses)
