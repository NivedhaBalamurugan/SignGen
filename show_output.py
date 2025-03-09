import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from config import *
import numpy as np
import matplotlib.pyplot as plt


def plot_a_frame(J, filename):
    J = np.array(J).reshape(49, 2)  
    if np.all(J == 0):
        return  

    J1 = J[:7, :]  
    J2 = J[7:, :]  

    hand_connections = [(0, 1), (1, 2), (2, 3), (3, 4),
                        (0, 5), (5, 6), (6, 7), (7, 8),
                        (5, 9), (9, 10), (10, 11), (11, 12),
                        (9, 13), (13, 14), (14, 15), (15, 16),
                        (13, 17), (17, 18), (18, 19), (19, 20)]

    upper_body_connections = [(0, 1), (1, 3), (0, 2)]  

    hand_to_body = [(21, 3), (0, 2)]  

    left_hand_valid = not np.all(J2[0] == 0)  
    right_hand_valid = not np.all(J2[21] == 0)  

    midpoint = (J1[0] + J1[1]) / 2    

    plt.figure(figsize=(6, 6))
    plt.gca().set_aspect('equal', adjustable='box')

    if left_hand_valid:
        plt.scatter(J2[:21, 0], J2[:21, 1], color='blue', s=10, label="Left Hand")
        for start, end in hand_connections:
            plt.plot([J2[start, 0], J2[end, 0]],
                    [J2[start, 1], J2[end, 1]],
                    color='green', linewidth=1)

    if right_hand_valid:
        plt.scatter(J2[21:, 0], J2[21:, 1], color='purple', s=10, label="Right Hand")
        for start, end in hand_connections:
            start += 21
            end += 21
            plt.plot([J2[start, 0], J2[end, 0]],
                    [J2[start, 1], J2[end, 1]],
                    color='orange', linewidth=1)

    body_indices_to_plot = [0, 1, 2, 3, 6]  
    plt.scatter(J1[body_indices_to_plot, 0], J1[body_indices_to_plot, 1], color='red', s=15, label="Body Landmarks")

    plt.plot([J1[6, 0], midpoint[0]], 
            [J1[6, 1], midpoint[1]], 
            color='black', linewidth=2, label="6th to Midpoint")

    for start, end in upper_body_connections:
        plt.plot([J1[start, 0], J1[end, 0]],
                [J1[start, 1], J1[end, 1]],
                color='black', linewidth=2)

    if left_hand_valid:
        plt.plot([J2[0, 0], J1[2, 0]],
                [J2[0, 1], J1[2, 1]],
                color='purple', linestyle="dashed", linewidth=1.5)

    if right_hand_valid:
        plt.plot([J2[21, 0], J1[3, 0]],
                [J2[21, 1], J1[3, 1]],
                color='purple', linestyle="dashed", linewidth=1.5)

    plt.gca().invert_yaxis()
    plt.title("2D Projected Hand & Body Landmarks with Connections")
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="upper right")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_a_frame_29_joints(J, filename):

    J = np.array(J).reshape(NUM_JOINTS, NUM_COORDINATES)  
    if np.all(J == 0):
        return  

    J1 = J[:7, :]  
    J2 = J[7:, :]  

    single_hand_connections = [
        (0, 1), (1, 2),      
        (0, 3), (3, 4),      
        (0, 5), (5, 6),      
        (0, 7), (7, 8),      
        (0, 9), (9, 10),     
    ]

    upper_body_connections = [(0, 1), (1, 3), (0, 2)]
    hand_to_body = [(11, 3), (0, 2)]

    midpoint = (J1[0] + J1[1]) / 2    
    left_hand_valid = not np.all(J2[0] == 0)    
    right_hand_valid = not np.all(J2[11] == 0)  

    plt.figure(figsize=(6, 6))
    plt.gca().set_aspect('equal', adjustable='box')

    if left_hand_valid:
        plt.scatter(J2[:11, 0], J2[:11, 1], color='blue', s=10, label="Left Hand")
        for start, end in single_hand_connections:
            plt.plot([J2[start, 0], J2[end, 0]],
                    [J2[start, 1], J2[end, 1]],
                    color='green', linewidth=1)

    if right_hand_valid:
        plt.scatter(J2[11:22, 0], J2[11:22, 1], color='purple', s=10, label="Right Hand")
        for start, end in single_hand_connections:
            start += 11
            end += 11
            plt.plot([J2[start, 0], J2[end, 0]],
                    [J2[start, 1], J2[end, 1]],
                    color='orange', linewidth=1)

    body_indices_to_plot = [0, 1, 2, 3, 6]  
    plt.scatter(J1[body_indices_to_plot, 0], J1[body_indices_to_plot, 1], color='red', s=15, label="Body Landmarks")

    for start, end in upper_body_connections:
        plt.plot([J1[start, 0], J1[end, 0]],
                [J1[start, 1], J1[end, 1]],
                color='black', linewidth=2)

    plt.plot([J1[6, 0], midpoint[0]], 
            [J1[6, 1], midpoint[1]], 
            color='black', linewidth=2, label="6th to Midpoint")

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
            continue  
        plot_a_frame_29_joints(frame, f"{frame_path}/frame_{valid_frame_count}.png")
        valid_frame_count += 1


