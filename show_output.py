import numpy as np
import matplotlib.pyplot as plt
import os
from config import *

def plot_a_frame(J, filename):
    J = np.array(J).reshape(49, 3)  # Reshape into (49, 3)
    J1 = J[:7, :]  # First 7 rows → (7, 3)
    J2 = J[7:, :]  # Remaining 42 rows → (42, 3)

    hand_connections = [(0, 1), (1, 2), (2, 3), (3, 4),
                        (0, 5), (5, 6), (6, 7), (7, 8),
                        (5, 9), (9, 10), (10, 11), (11, 12),
                        (9, 13), (13, 14), (14, 15), (15, 16),
                        (13, 17), (17, 18), (18, 19), (19, 20)]

    upper_body_connections = [(0,1),(1,3),(0,2),(0,4), (1,5), (4,5)]

    hand_to_body = [(21,3), (0,2)]

    J2 = np.array(J2)
    J1 = np.array(J1)
    J2_proj = J2[:, :2]
    J1_proj = J1[:, :2]
    J2_proj[:, 1] = -J2_proj[:, 1]
    J1_proj[:, 1] = -J1_proj[:, 1]
    midpoint = (J1_proj[0] + J1_proj[1]) / 2

    plt.figure(figsize=(6, 6))

    plt.gca().set_aspect('equal', adjustable='box')

    plt.scatter(J2_proj[:21, 0], J2_proj[:21, 1], color='blue', s=10, label="Left Hand")
    plt.scatter(J2_proj[21:, 0], J2_proj[21:, 1], color='purple', s=10, label="Right Hand")
    plt.scatter(J1_proj[:, 0], J1_proj[:, 1], color='red', s=15, label="Body Landmarks")
    plt.scatter(midpoint[0], midpoint[1], color='cyan', s=15, label="Midpoint")

    for start, end in hand_connections:
        plt.plot([J2_proj[start, 0], J2_proj[end, 0]],
                [J2_proj[start, 1], J2_proj[end, 1]], 
                color='green', linewidth=1)

    for start, end in hand_connections:
        start += 21
        end += 21
        plt.plot([J2_proj[start, 0], J2_proj[end, 0]],
                [J2_proj[start, 1], J2_proj[end, 1]], 
                color='orange', linewidth=1)

    for start, end in upper_body_connections:
        plt.plot([J1_proj[start, 0], J1_proj[end, 0]],
                [J1_proj[start, 1], J1_proj[end, 1]], 
                color='black', linewidth=2)

    for hand_idx, body_idx in hand_to_body:
        plt.plot([J2_proj[hand_idx, 0], J1_proj[body_idx, 0]],
                [J2_proj[hand_idx, 1], J1_proj[body_idx, 1]], 
                color='purple', linestyle="dashed", linewidth=1.5)

    plt.plot([J1_proj[6, 0], midpoint[0]], 
            [J1_proj[6, 1], midpoint[1]], 
            color='black', linewidth=2, label="6th to Midpoint")

    plt.title("Projected Hand & Body Landmarks with Connections")
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def images_to_video_ffmpeg(image_folder, output_video, fps=30):
    os.system(f"ffmpeg -framerate {fps} -i {image_folder}/frame_%d.png -c:v libx264 -pix_fmt yuv420p {output_video}")


def save_generated_sequence(generated_sequence, frame_path, video_path):

    output_dir = frame_path
    os.makedirs(frame_path, exist_ok=True)

    for i, frame in enumerate(generated_sequence):
        filename = os.path.join(output_dir, f"frame_{i}.png")
        plot_a_frame(frame, filename)
    
    logging.info(f"Saved {len(generated_sequence)} frames in '{output_dir}'")

    images_to_video_ffmpeg(frame_path, video_path, fps=30)
    logging.info(f"Video saved to {video_path}")
   
