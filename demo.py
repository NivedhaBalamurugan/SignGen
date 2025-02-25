# import cv2
# import numpy as np
# import mediapipe as mp
# import json
# import os
# from tqdm import tqdm
# import data_processing


# NUM_LANDMARKS = 21 + 21 + 6 + 1

# mp_hands = mp.solutions.hands
# mp_pose = mp.solutions.pose

# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
# pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# def process_frame(frame):
#     frame = cv2.resize(frame, (224, 224))
#     return frame

# def normalize_landmarks(landmarks, frame_width=224, frame_height=224):
#     landmarks[:, 0] /= frame_width  
#     landmarks[:, 1] /= frame_height  
#     return landmarks

# def denormalize(landmarks, frame_width=224, frame_height=224):
#     landmarks[:, 0] *= frame_width  
#     landmarks[:, 1] *= frame_height  
#     return landmarks

# def get_frame_landmarks(frame):
#     palm_landmarks = np.zeros((42, 3), dtype=np.float64)
#     body_landmarks = np.zeros((7, 3), dtype=np.float64)

#     results_hands = hands.process(frame)
#     if results_hands.multi_hand_landmarks:
#         for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
#             if i < 2:
#                 offset = 21 * i
#                 for idx, landmark in enumerate(hand_landmarks.landmark):
#                     if idx < 21:
#                         palm_landmarks[offset + idx] = [landmark.x, landmark.y, landmark.z]
#                         # Draw smaller landmarks on the frame
#                         h, w, _ = frame.shape
#                         cx, cy = int(landmark.x * w), int(landmark.y * h)
#                         cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)  # Green dots for hands (size reduced)

#     results_pose = pose.process(frame)
#     if results_pose.pose_landmarks:
#         upper_body_indices = [11, 12, 13, 14, 23, 24]
#         for i, idx in enumerate(upper_body_indices):
#             landmark = results_pose.pose_landmarks.landmark[idx]
#             body_landmarks[i] = [landmark.x, landmark.y, landmark.z]
#             # Draw smaller pose landmarks
#             h, w, _ = frame.shape
#             cx, cy = int(landmark.x * w), int(landmark.y * h)
#             cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)  # Blue dots for upper body (size reduced)

#         face_landmark = results_pose.pose_landmarks.landmark[0]  
#         body_landmarks[6] = [face_landmark.x, face_landmark.y, face_landmark.z]
#         # Draw smaller face landmark
#         cx, cy = int(face_landmark.x * w), int(face_landmark.y * h)
#         cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)  # Red dot for face (size reduced)

#     # Display the frame with keypoints
#     cv2.imshow("Frame with Keypoints", frame)
#     cv2.waitKey(1)  # Small delay to update the window

#     return palm_landmarks, body_landmarks





# def get_video_landmarks(videoPath, start_frame, end_frame):
#     cap = cv2.VideoCapture(video_path)
#     original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Define a reasonable display size (keep original if small, resize if too large)
#     MAX_WIDTH, MAX_HEIGHT = 1280, 720  # Adjust these as needed

#     if original_width > MAX_WIDTH or original_height > MAX_HEIGHT:
#         scale = min(MAX_WIDTH / original_width, MAX_HEIGHT / original_height)
#         display_width = int(original_width * scale)
#         display_height = int(original_height * scale)
#     else:
#         display_width, display_height = original_width, original_height

#     cv2.namedWindow("Frame with Keypoints", cv2.WINDOW_NORMAL)  # Allow resizing
#     cv2.resizeWindow("Frame with Keypoints", display_width, display_height)  # Set optimal size
#     if start_frame < 1:
#         start_frame = 1
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if end_frame < 0 or end_frame > total_frames:
#         end_frame = total_frames

#     all_palm_landmarks = []
#     all_body_landmarks = []

#     for frame_index in range(1, total_frames + 1):
#         res, frame = cap.read()
#         if not res:
#             break
#         if start_frame <= frame_index <= end_frame:
#             frame.flags.writeable = False
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = process_frame(frame)

#             palm_landmarks, body_landmarks = get_frame_landmarks(frame)
#             palm_landmarks = normalize_landmarks(palm_landmarks)
#             body_landmarks = normalize_landmarks(body_landmarks)

#             all_palm_landmarks.append(palm_landmarks.tolist())
#             all_body_landmarks.append(body_landmarks.tolist())

#     cap.release()
#     return all_palm_landmarks, all_body_landmarks



# data = data_processing.processed_data[10]  # Process only the first video
# video_dir = "Dataset/videos"
# video_path_to_find = os.path.join(video_dir, "07070.mp4")  # Correct video path format

# matching_data = next((data for data in data_processing.processed_data if data["video_path"] == video_path_to_find), None)

# video_path = matching_data["video_path"]
# start_frame = matching_data["frame_start"]
# end_frame = matching_data["frame_end"]
# gloss = matching_data["gloss"]

# # output_dir = "output_frames"
# # os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# # output_unprocessed = "output_unprocessed"
# # output_color_convert = "output_color_convert"

# frame_count = 0  # Track frame number

# try:
#     cap = cv2.VideoCapture(video_path)

#     while cap.isOpened():
#         res, frame = cap.read()
#         if not res:
#             break

#         frame = process_frame(frame)
#         # cv2.imshow("Frame", frame)
#         #frame_filename = os.path.join(output_unprocessed, f"frame_{frame_count:04d}.png")
#         #cv2.imwrite(frame_filename, frame)
#         # frame_count += 1

#         # Convert frame for MediaPipe
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # cv2.imshow("Frame", rgb_frame)
#         #frame_filename = os.path.join(output_color_convert, f"frame_{frame_count:04d}.png")
#         #cv2.imwrite(frame_filename, frame)
#         # frame_count += 1

       

#         # Extract and draw landmarks
#         palm_landmarks, body_landmarks = get_frame_landmarks(frame)

#         if frame_count == 41:
#             frame_41_palm = denormalize(palm_landmarks)
#             frame_41_body = denormalize(body_landmarks)
#             print("got")
#             break

#         # Convert back to BGR before displaying
#         #display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
#         # Show the frame
#         #cv2.imshow("Frame with Keypoints", display_frame)

#         #Save the frame locally
#         # frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
#         # cv2.imwrite(frame_filename, frame)
#         frame_count += 1

#         if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#    # print(f"Frames saved in {output_dir}")

# except Exception as e:
#     print(f"Error processing video {video_path}: {e}")


# print("palm")
# print(frame_41_palm)
# print("body")
# print(frame_41_body)


import numpy as np
import matplotlib.pyplot as plt
import augmentation

# #J1 = body J2 = palm

J1 = [[ 1.54245346e+02 , 1.21273857e+02, -4.19018894e-01],
 [ 7.92570877e+01 , 1.18067162e+02, -3.54202956e-01],
 [ 1.54481319e+02 , 2.13377394e+02 ,-7.73395360e-01],
 [ 7.47275448e+01 , 2.11177837e+02 ,-7.04671204e-01],
 [ 1.40013432e+02 , 2.32800522e+02 ,-1.67252272e-02],
 [ 9.43878412e+01 , 2.31823143e+02,  1.90149434e-02],
 [ 1.20822151e+02 , 7.69493580e+01, -1.02578211e+00]]


J2 =[[ 1.30074045e+02 , 1.56240585e+02 ,-9.78548442e-08],
 [ 1.33344080e+02 , 1.44369913e+02, -7.20441341e-03],
 [ 1.36751888e+02 , 1.33177307e+02, -1.15272747e-02],
 [ 1.40681618e+02 , 1.24886770e+02, -1.56324580e-02],
 [ 1.46186831e+02 , 1.20013735e+02, -2.11218204e-02],
 [ 1.36296885e+02 , 1.40944775e+02, -1.37533043e-02],
 [ 1.34963476e+02 , 1.34919243e+02, -2.32123081e-02],
 [ 1.33804237e+02 , 1.31242123e+02, -3.21619995e-02],
 [ 1.32824909e+02 , 1.27007887e+02, -3.99082229e-02],
 [ 1.32509148e+02 , 1.44722631e+02, -1.72255058e-02],
 [ 1.31014748e+02 , 1.38062572e+02, -2.27133259e-02],
 [ 1.29804907e+02 , 1.33752821e+02, -2.85596289e-02],
 [ 1.28537922e+02 , 1.29637840e+02, -3.70936282e-02],
 [ 1.27843033e+02 , 1.47704796e+02, -2.15148088e-02],
 [ 1.26984615e+02 , 1.42227020e+02, -2.84827556e-02],
 [ 1.25940626e+02 , 1.38102146e+02, -3.05977967e-02],
 [ 1.24977266e+02 , 1.33888285e+02, -3.51727903e-02],
 [ 1.22504072e+02 , 1.50291157e+02, -2.65672151e-02],
 [ 1.21889519e+02 , 1.46372362e+02, -3.33095863e-02],
 [ 1.21968266e+02 , 1.43502256e+02, -3.43618281e-02],
 [ 1.21319799e+02 , 1.40643005e+02, -3.72319594e-02],
 [ 1.04953363e+02 , 1.54218443e+02,  5.72060017e-08],
 [ 1.00883237e+02 , 1.46574183e+02, -1.27955051e-02],
 [ 9.74039650e+01 , 1.37309658e+02, -2.26288661e-02],
 [ 9.41783104e+01 , 1.30018677e+02, -3.07885930e-02],
 [ 8.98777914e+01 , 1.25134586e+02, -3.79496031e-02],
 [ 1.00064967e+02 , 1.43062553e+02, -2.52817143e-02],
 [ 1.01156260e+02 , 1.38937346e+02, -4.01303843e-02],
 [ 1.03542390e+02 , 1.35325447e+02, -4.96741496e-02],
 [ 1.05704208e+02 , 1.32394192e+02, -5.59597127e-02],
 [ 1.05204090e+02 , 1.46702183e+02, -2.64722500e-02],
 [ 1.06498058e+02 , 1.43051352e+02, -3.82856280e-02],
 [ 1.08874775e+02 , 1.38975864e+02 ,-4.27568145e-02],
 [ 1.10438689e+02 , 1.35900814e+02, -4.71030511e-02],
 [ 1.10265621e+02 , 1.50049202e+02, -2.84748413e-02],
 [ 1.11250857e+02 , 1.46726522e+02, -4.06831019e-02],
 [ 1.13249962e+02 , 1.43014608e+02, -3.99910174e-02],
 [ 1.14236380e+02 , 1.40201019e+02, -3.88027132e-02],
 [ 1.15029509e+02 , 1.52785379e+02, -3.13975401e-02],
 [ 1.16769308e+02 , 1.48565323e+02, -3.98570634e-02],
 [ 1.17655977e+02 , 1.45027966e+02, -3.84582765e-02],
 [ 1.18099766e+02 , 1.41934050e+02, -3.72201838e-02]]




hand_connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
                    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
                    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
                    (13, 17), (17, 18), (18, 19), (19, 20)]  # Little finger

upper_body_connections = [(0,1),(1,3),(0,2),(0,4), (1,5), (4,5)] # Add extra shoulder or neck joints if needed

hand_to_body = [(21,3), (0,2) ]

J2 = np.array(J2)  # Hand keypoints for frame 22 (both left and right hands)
J1 = np.array(J1)  # Body keypoints for frame 22

# augJ2 = data_aug.augment_skeleton_sequence(J2)
# augJ1 = data_aug.augment_skeleton_sequence(J1)


# print("original")
# print(J1)

# print("aug")
# print(augJ1)

J2_proj = J2[:, :2]
J1_proj = J1[:, :2]

# augJ2_proj = augJ2[:, :2]
# augJ1_proj = augJ1[:, :2]

J2_proj[:, 1] = -J2_proj[:, 1]
J1_proj[:, 1] = -J1_proj[:, 1]

# augJ2_proj[:, 1] = -augJ2_proj[:, 1]
# augJ1_proj[:, 1] = -augJ1_proj[:, 1]


# # Midpoints
# midpoint_orig = (J1_proj[0] + J1_proj[1]) / 2
# midpoint_aug = (augJ1_proj[0] + augJ1_proj[1]) / 2

# # Create the figure
# plt.figure(figsize=(8, 8))
# plt.gca().set_aspect('equal', adjustable='box')

# # ============ PLOT ORIGINAL LANDMARKS ============
# plt.scatter(J2_proj[:21, 0], J2_proj[:21, 1], color='blue', s=10, label="Original Left Hand")  
# plt.scatter(J2_proj[21:, 0], J2_proj[21:, 1], color='red', s=10, label="Original Right Hand")  
# plt.scatter(J1_proj[:, 0], J1_proj[:, 1], color='black', s=15, label="Original Body")

# # ============ PLOT AUGMENTED LANDMARKS ============
# plt.scatter(augJ2_proj[:21, 0], augJ2_proj[:21, 1], color='purple', s=10, label="Augmented Left Hand")  
# plt.scatter(augJ2_proj[21:, 0], augJ2_proj[21:, 1], color='orange', s=10, label="Augmented Right Hand")  
# plt.scatter(augJ1_proj[:, 0], augJ1_proj[:, 1], color='green', s=15, label="Augmented Body")

# # ============ PLOT CONNECTIONS (Original) ============
# for start, end in hand_connections:
#     plt.plot([J2_proj[start, 0], J2_proj[end, 0]], 
#              [J2_proj[start, 1], J2_proj[end, 1]], 
#              color='green', linewidth=1)

# for start, end in hand_connections:
#     start += 21
#     end += 21
#     plt.plot([J2_proj[start, 0], J2_proj[end, 0]], 
#              [J2_proj[start, 1], J2_proj[end, 1]], 
#              color='blue', linewidth=1)

# for start, end in upper_body_connections:
#     plt.plot([J1_proj[start, 0], J1_proj[end, 0]], 
#              [J1_proj[start, 1], J1_proj[end, 1]], 
#              color='black', linewidth=2)

# for hand_idx, body_idx in hand_to_body:
#     plt.plot([J2_proj[hand_idx, 0], J1_proj[body_idx, 0]],  
#              [J2_proj[hand_idx, 1], J1_proj[body_idx, 1]], 
#              color='purple', linestyle="dashed", linewidth=1.5)

# # ============ PLOT CONNECTIONS (Augmented) ============
# for start, end in hand_connections:
#     plt.plot([augJ2_proj[start, 0], augJ2_proj[end, 0]], 
#              [augJ2_proj[start, 1], augJ2_proj[end, 1]], 
#              color='darkviolet', linewidth=1)

# for start, end in hand_connections:
#     start += 21
#     end += 21
#     plt.plot([augJ2_proj[start, 0], augJ2_proj[end, 0]], 
#              [augJ2_proj[start, 1], augJ2_proj[end, 1]], 
#              color='orange', linewidth=1)

# for start, end in upper_body_connections:
#     plt.plot([augJ1_proj[start, 0], augJ1_proj[end, 0]], 
#              [augJ1_proj[start, 1], augJ1_proj[end, 1]], 
#              color='green', linewidth=2)

# for hand_idx, body_idx in hand_to_body:
#     plt.plot([augJ2_proj[hand_idx, 0], augJ1_proj[body_idx, 0]],  
#              [augJ2_proj[hand_idx, 1], augJ1_proj[body_idx, 1]], 
#              color='darkred', linestyle="dashed", linewidth=1.5)

# # Midpoint Connection
# plt.plot([J1_proj[6, 0], midpoint_orig[0]], 
#          [J1_proj[6, 1], midpoint_orig[1]], 
#          color='black', linewidth=2, label="Original 6th to Midpoint")

# plt.plot([augJ1_proj[6, 0], midpoint_aug[0]], 
#          [augJ1_proj[6, 1], midpoint_aug[1]], 
#          color='green', linewidth=2, label="Augmented 6th to Midpoint")

# # ============ AXES SETTINGS ============
# plt.title("Original vs Augmented Hand & Body Landmarks")
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.grid(True)  # Show grid for better clarity
# plt.show()


# plt.figure(figsize=(6, 6))
# plt.scatter(J1_proj[:, 0], J1_proj[:, 1], color='red', s=15, label="Original Body Landmarks")
# plt.title("Original Body Landmarks")
# plt.axis("equal")
# plt.xticks([])
# plt.yticks([])
# plt.legend()
# plt.show()

# # Plot augmented landmarks
# plt.figure(figsize=(6, 6))
# plt.scatter(augJ1_proj[:, 0], augJ1_proj[:, 1], color='blue', s=15, label="Augmented Body Landmarks")
# plt.title("Augmented Body Landmarks")
# plt.axis("equal")
# plt.xticks([])
# plt.yticks([])
# plt.legend()
# plt.show()


midpoint = (J1_proj[0] + J1_proj[1]) / 2

plt.figure(figsize=(6, 6))  # Adjust the figure size to maintain the aspect ratio

plt.gca().set_aspect('equal', adjustable='box')


plt.scatter(J2_proj[:21, 0], J2_proj[:21, 1], color='blue', s=10, label="Left Hand")  # Left Hand
plt.scatter(J2_proj[21:, 0], J2_proj[21:, 1], color='purple', s=10, label="Right Hand")  # Right Hand
plt.scatter(J1_proj[:, 0], J1_proj[:, 1], color='red', s=15, label="Body Landmarks")
plt.scatter(midpoint[0], midpoint[1], color='cyan', s=15, label="Midpoint")  # Optional: Show midpoint

# Draw left-hand connections
for start, end in hand_connections:
    plt.plot([J2_proj[start, 0], J2_proj[end, 0]],
             [J2_proj[start, 1], J2_proj[end, 1]], 
             color='green', linewidth=1)

# Draw right-hand connections (offset indices by +21)
for start, end in hand_connections:
    start += 21
    end += 21
    plt.plot([J2_proj[start, 0], J2_proj[end, 0]],
             [J2_proj[start, 1], J2_proj[end, 1]], 
             color='orange', linewidth=1)

# Draw body connections
for start, end in upper_body_connections:
    plt.plot([J1_proj[start, 0], J1_proj[end, 0]],
             [J1_proj[start, 1], J1_proj[end, 1]], 
             color='black', linewidth=2)

# Draw hand-to-body connections correctly
for hand_idx, body_idx in hand_to_body:
    plt.plot([J2_proj[hand_idx, 0], J1_proj[body_idx, 0]],  # Hand index from J2, body index from J1
             [J2_proj[hand_idx, 1], J1_proj[body_idx, 1]], 
             color='purple', linestyle="dashed", linewidth=1.5)

plt.plot([J1_proj[6, 0], midpoint[0]], 
         [J1_proj[6, 1], midpoint[1]], 
         color='black', linewidth=2, label="6th to Midpoint")

plt.title("Projected Hand & Body Landmarks with Connections")
plt.axis("equal")
plt.xticks([])
plt.yticks([])
plt.legend()
plt.show()





