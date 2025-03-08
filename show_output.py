import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from config import *

import numpy as np
import matplotlib.pyplot as plt


def plot_a_frame_29_joints(J, filename):
    """
    Plot a 2D frame of body and hand landmarks with connections.
    
    :param J: List or numpy array of shape (36, 2) representing 2D joint coordinates.
    :param filename: Output filename to save the plot.
    """
    J = np.array(J).reshape(NUM_JOINTS, NUM_COORDINATES)  # Ensure shape is correct
    if np.all(J == 0):
        return  # Skip entirely blank frames

    J1 = J[:7, :]  # Body joints (7 joints)
    J2 = J[7:, :]  # Hand joints (29 joints)

    # Define connections for each hand (based on new structure)
    # For one hand: 0=palm, (1,2)=finger1, (3,4)=finger2, (5,6)=finger3, (7,8)=finger4, (9,10)=finger5
    single_hand_connections = [
        (0, 1), (1, 2),      # Finger 1
        (0, 3), (3, 4),      # Finger 2
        (0, 5), (5, 6),      # Finger 3
        (0, 7), (7, 8),      # Finger 4
        (0, 9), (9, 10),     # Finger 5
    ]
    
    # Define body connections
    upper_body_connections = [(0, 1), (1, 3), (0, 2), (0, 4), (1, 5), (4, 5)]
    
    # Hand to body connections
    # body index 3 connected to hand index 11 (second hand palm)
    # body index 2 connected to hand index 0 (first hand palm)
    hand_to_body = [(11, 3), (0, 2)]

    midpoint = (J1[0] + J1[1]) / 2    

    # Check if both hands should be drawn
    left_hand_valid = not np.all(J2[0] == 0)    # Palm of left hand (index 0)
    right_hand_valid = not np.all(J2[11] == 0)  # Palm of right hand (index 11)

    plt.figure(figsize=(6, 6))
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot left hand
    if left_hand_valid:
        plt.scatter(J2[:11, 0], J2[:11, 1], color='blue', s=10, label="Left Hand")
        for start, end in single_hand_connections:
            plt.plot([J2[start, 0], J2[end, 0]],
                     [J2[start, 1], J2[end, 1]],
                     color='green', linewidth=1)
    
    # Plot right hand
    if right_hand_valid:
        plt.scatter(J2[11:22, 0], J2[11:22, 1], color='purple', s=10, label="Right Hand")
        for start, end in single_hand_connections:
            start += 11
            end += 11
            plt.plot([J2[start, 0], J2[end, 0]],
                     [J2[start, 1], J2[end, 1]],
                     color='orange', linewidth=1)
    
    # Plot body landmarks
    plt.scatter(J1[:, 0], J1[:, 1], color='red', s=15, label="Body Landmarks")
    
    # Plot body connections
    for start, end in upper_body_connections:
        plt.plot([J1[start, 0], J1[end, 0]],
                 [J1[start, 1], J1[end, 1]],
                 color='black', linewidth=2)

    # Connect nose (body index 0) to midpoint of shoulders
    plt.plot([J1[6, 0], midpoint[0]], 
         [J1[6, 1], midpoint[1]], 
         color='black', linewidth=2, label="6th to Midpoint")
    
    # Plot hand-to-body connections
    if left_hand_valid:
        plt.plot([J2[0, 0], J1[2, 0]],
                 [J2[0, 1], J1[2, 1]],
                 color='purple', linestyle="dashed", linewidth=1.5)
    
    if right_hand_valid:
        plt.plot([J2[11, 0], J1[3, 0]],
                 [J2[11, 1], J1[3, 1]],
                 color='purple', linestyle="dashed", linewidth=1.5)
    
    plt.gca().invert_yaxis()
    plt.title("2D Projected Hand & Body Landmarks with Connections")
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="upper right")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def images_to_video_ffmpeg(image_folder, output_video, fps=30):
    os.system(f"ffmpeg -framerate {fps} -i {image_folder}/frame_%d.png -c:v libx264 -pix_fmt yuv420p {output_video}")


def save_generated_sequence(generated_sequence, frame_path, video_path):
    os.makedirs(frame_path, exist_ok=True)
    valid_frame_count = 0

    for i, frame in enumerate(generated_sequence):
        frame_array = np.array(frame)
        if np.all(frame_array == 0):
            continue  # Skip all-zero frames
        plot_a_frame_29_joints(frame, f"{frame_path}/frame_{valid_frame_count}.png")
        valid_frame_count += 1

    logging.info(f"Saved {valid_frame_count} frames in '{frame_path}'")
    images_to_video_ffmpeg(frame_path, video_path)
    logging.info(f"Video saved to {video_path}")
