import os
import logging
from config import *
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
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

def plot_a_frame_29_joints(J, filename, x_min, x_max, y_min, y_max, pre_defined_body_values=True):
    
    # from matplotlib.backends.backend_agg import FigureCanvasAgg
    # fig = plt.figure(figsize=(6, 6))
    # canvas = FigureCanvasAgg(fig)
    # ax = fig.add_axes([0, 0, 1, 1])
    try:
        J = np.array(J)
        if np.all(J == 0):
            return  # Skip entirely blank frames

        J1 = J[:7, :]   # Body joints (7 joints)
        J2 = J[7:, :]   # Hand joints (22 joints)

        if pre_defined_body_values:
            J1[0] = RIGHT_SHOULDER_VALUE
            J1[1] = LEFT_SHOULDER_VALUE
            J1[4] = RIGHT_HIP_VALUE
            J1[5] = LEFT_HIP_VALUE
            J1[6] = NOSE_VALUE

        single_hand_connections = [
            (0, 1), (1, 2),      # Finger 1
            (0, 3), (3, 4),      # Finger 2
            (0, 5), (5, 6),      # Finger 3
            (0, 7), (7, 8),      # Finger 4
            (0, 9), (9, 10),     # Finger 5
        ]

        upper_body_connections = [(0, 1), (1, 3), (0, 2), (0, 4), (1, 5)]

        fig = plt.figure(figsize=(6, 6), clear=True)
        ax = fig.add_axes([0, 0, 1, 1])  
        ax.set_axis_off()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)

        def should_plot(joint):
            return not (joint[0] == 0 and joint[1] == 0)

        def is_valid_hand_joint(joint, threshold=1e-2):
            return np.linalg.norm(joint) >= threshold

        # Plot left hand (0 to 10 in J2)
        left_hand_to_plot = [i for i in range(11) if is_valid_hand_joint(J2[i])]
        ax.scatter(J2[left_hand_to_plot, 0], J2[left_hand_to_plot, 1], color='blue', s=10, label="Left Hand")
        for start, end in single_hand_connections:
            if is_valid_hand_joint(J2[start]) and is_valid_hand_joint(J2[end]):
                ax.plot([J2[start, 0], J2[end, 0]],
                        [J2[start, 1], J2[end, 1]],
                        color='green', linewidth=1)

        # Plot right hand (11 to 21 in J2)
        right_hand_to_plot = [i for i in range(11, 22) if is_valid_hand_joint(J2[i])]
        ax.scatter(J2[right_hand_to_plot, 0], J2[right_hand_to_plot, 1], color='purple', s=10, label="Right Hand")
        for start, end in single_hand_connections:
            start += 11
            end += 11
            if is_valid_hand_joint(J2[start]) and is_valid_hand_joint(J2[end]):
                ax.plot([J2[start, 0], J2[end, 0]],
                        [J2[start, 1], J2[end, 1]],
                        color='orange', linewidth=1)

        # Plot body landmarks
        body_indices_to_plot = [i for i in [0, 1, 2, 3, 6] if should_plot(J1[i])]
        ax.scatter(J1[body_indices_to_plot, 0], J1[body_indices_to_plot, 1], color='red', s=15, label="Body Landmarks")

        # Plot body connections
        for start, end in upper_body_connections:
            if should_plot(J1[start]) and should_plot(J1[end]):
                ax.plot([J1[start, 0], J1[end, 0]],
                        [J1[start, 1], J1[end, 1]],
                        color='black', linewidth=2)

        # Connect nose to midpoint of shoulders
        if should_plot(J1[6]) and should_plot(J1[0]) and should_plot(J1[1]):
            midpoint = (J1[0] + J1[1]) / 2
            ax.plot([J1[6, 0], midpoint[0]],
                    [J1[6, 1], midpoint[1]],
                    color='black', linewidth=2, label="6th to Midpoint")

        # Connect hands to body
        if len(J2) > 0 and is_valid_hand_joint(J2[0]) and should_plot(J1[2]):
            ax.plot([J2[0, 0], J1[2, 0]],
                    [J2[0, 1], J1[2, 1]],
                    color='purple', linestyle="dashed", linewidth=1.5)

        if len(J2) > 11 and is_valid_hand_joint(J2[11]) and should_plot(J1[3]):
            ax.plot([J2[11, 0], J1[3, 0]],
                    [J2[11, 1], J1[3, 1]],
                    color='purple', linestyle="dashed", linewidth=1.5)

        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close('all')
    except Exception as e:
        logging.error(f"Error plotting frame: {e}")
    finally:
        plt.close(fig)

def images_to_video_ffmpeg(image_folder, output_video, fps=7):
    os.system(f"ffmpeg -framerate {fps} -i {image_folder}/frame_%d.png -c:v libx264 -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" {output_video}")

def calculate_global_limits(sequence):
    all_joints = np.concatenate([frame for frame in sequence if not np.all(frame == 0)])
    x_coords = all_joints[:, 0]
    y_coords = all_joints[:, 1]
    
    x_center = (np.max(x_coords) + np.min(x_coords)) / 2
    y_center = (np.max(y_coords) + np.min(y_coords)) / 2
    
    x_range = (np.max(x_coords) - np.min(x_coords)) * 1.2
    y_range = (np.max(y_coords) - np.min(y_coords)) * 1.2
    
    limit = max(x_range, y_range) / 2
    
    return (x_center - limit, x_center + limit, 
            y_center - limit, y_center + limit)


def delete_files(path):
    if os.path.isdir(path):
        for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  
                except Exception as e:
                    logging.error(f"Failed to delete {file_path}. Reason: {e}")
    elif os.path.isfile(path):
        try:
            os.unlink(path)  
        except Exception as e:
            logging.error(f"Failed to delete {path}. Reason: {e}")
    

def save_generated_sequence(generated_sequence, frame_path, video_path):
    os.makedirs(frame_path, exist_ok=True)
    valid_frame_count = 0

    delete_files(frame_path)
    delete_files(video_path)
    
    x_min, x_max, y_min, y_max = calculate_global_limits(generated_sequence)

    for i, frame in enumerate(generated_sequence):
        frame_array = np.array(frame)
        if np.all(frame_array == 0):
            continue  
        if(NUM_JOINTS == 49):
            plot_a_frame(frame, f"{frame_path}/frame_{valid_frame_count}.png")
        else:
            plot_a_frame_29_joints(frame, f"{frame_path}/frame_{valid_frame_count}.png", x_min, x_max, y_min, y_max)
        valid_frame_count += 1

    logging.info(f"Saved {valid_frame_count} frames")
    images_to_video_ffmpeg(frame_path, video_path)
    logging.info(f"Saved video to {video_path}")
    


