import json
import numpy as np  # Ensure NumPy is imported
import show_output
from config import *
from utils.data_utils import select_sign_frames


# def read_jsonl_file(file_path):
#     data = {}
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             item = json.loads(line.strip())
#             data.update(item)
#     return data

# def get_video(data, key):
#     if key in data and len(data[key]) >= 2:
#         return data[key][0]  # Get the second video (index 1)
#     else:
#         return None

# # Path to your JSONL file
# file_path = 'Dataset/landmarks/final/0_landmarks_top20_split1_aug.jsonl' 

# # Read the JSONL file
# data = read_jsonl_file(file_path)

# # Get the second video for the key "afternoon"
# video_data = get_video(data, 'after')

# if video_data is not None:
#     aug_video = np.array(video_data)  # Convert only if video_data is not None
#     print("Second video for 'afternoon':", aug_video.shape)

#     # print(aug_video[1])
#     key_frames = select_sign_frames(aug_video)
#     show_output.save_generated_sequence(key_frames, CVAE_OUTPUT_FRAMES, CVAE_OUTPUT_VIDEO)
# else:
#     print("Key 'afternoon' not found or does not have at least two videos.")


seq = [[[0.00321,0.00239],[0.00224,0.00238],[0.00332,0.00364],[0.00215,0.00354],[0.003,0.00431],[0.00239,0.00428],[0.00256,0.00117],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]],[[0.00322,0.00241],[0.00231,0.00243],[0.00333,0.00372],[0.00221,0.00357],[0.00306,0.0045],[0.00235,0.00446],[0.00261,0.00118],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]],[[0.00325,0.00246],[0.00231,0.00244],[0.00337,0.00381],[0.00221,0.0036],[0.00305,0.00465],[0.00243,0.00444],[0.0026,0.00118],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]],[[0.00327,0.00246],[0.00224,0.00249],[0.00337,0.00393],[0.00216,0.00389],[0.00303,0.0046],[0.00233,0.00473],[0.00261,0.00118],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]],[[0.0033,0.00244],[0.0022,0.00251],[0.0034,0.00391],[0.00212,0.00409],[0.00324,0.00465],[0.00238,0.00472],[0.00261,0.00118],[0.00177,0.00347],[0.00168,0.00337],[0.00161,0.00321],[0.00157,0.00308],[0.00155,0.00298],[0.00173,0.00311],[0.00178,0.00295],[0.00182,0.00285],[0.00186,0.00277],[0.0018,0.00312],[0.00185,0.00297],[0.00188,0.00286],[0.0019,0.00277],[0.00187,0.00314],[0.00191,0.003],[0.00194,0.0029],[0.00195,0.00281],[0.00193,0.00318],[0.00196,0.00305],[0.00197,0.00297],[0.00198,0.0029],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]],[[0.0033,0.00245],[0.00215,0.0025],[0.0034,0.00395],[0.00209,0.00415],[0.00322,0.00471],[0.00235,0.00474],[0.00262,0.00118],[0.0018,0.00312],[0.0017,0.00304],[0.00162,0.0029],[0.00159,0.00277],[0.00157,0.00267],[0.00176,0.00279],[0.00182,0.00263],[0.00186,0.00253],[0.0019,0.00246],[0.00184,0.0028],[0.0019,0.00265],[0.00193,0.00254],[0.00196,0.00246],[0.00191,0.00283],[0.00195,0.00268],[0.00198,0.00258],[0.002,0.00251],[0.00196,0.00286],[0.00199,0.00274],[0.00202,0.00265],[0.00203,0.00259],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]],[[0.00331,0.00244],[0.00213,0.00248],[0.00341,0.00393],[0.00206,0.00407],[0.0032,0.0047],[0.00231,0.00473],[0.00262,0.00119],[0.00181,0.00285],[0.00171,0.00277],[0.00166,0.00264],[0.00163,0.00253],[0.00163,0.00242],[0.00179,0.00251],[0.00186,0.00239],[0.00192,0.00232],[0.00197,0.00227],[0.00186,0.00253],[0.00193,0.00242],[0.00198,0.00235],[0.00202,0.00229],[0.00192,0.00257],[0.00198,0.00247],[0.00203,0.0024],[0.00206,0.00234],[0.00198,0.00261],[0.00203,0.00252],[0.00206,0.00246],[0.00209,0.00242],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]],[[0.00332,0.00244],[0.00211,0.00245],[0.00341,0.0039],[0.00205,0.00398],[0.0032,0.00469],[0.00235,0.00466],[0.00262,0.00119],[0.00185,0.0028],[0.00178,0.00266],[0.00175,0.00251],[0.00172,0.00238],[0.00169,0.00229],[0.00179,0.00245],[0.00186,0.00233],[0.00194,0.00226],[0.00201,0.00222],[0.00185,0.00248],[0.00192,0.00235],[0.002,0.00228],[0.00207,0.00223],[0.00191,0.00251],[0.00199,0.00238],[0.00206,0.00232],[0.00211,0.00227],[0.00199,0.00255],[0.00204,0.00244],[0.00209,0.00239],[0.00213,0.00234],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]],[[0.00331,0.00243],[0.00209,0.00244],[0.00342,0.00389],[0.00203,0.00399],[0.00321,0.00466],[0.00235,0.00462],[0.00263,0.00119],[0.00203,0.00263],[0.00194,0.00249],[0.0019,0.00234],[0.00188,0.00222],[0.00184,0.00215],[0.00203,0.00223],[0.00206,0.00217],[0.00206,0.00222],[0.00205,0.00227],[0.0021,0.00225],[0.00213,0.00221],[0.00212,0.00226],[0.00211,0.00231],[0.00217,0.00229],[0.0022,0.00227],[0.00218,0.00232],[0.00215,0.00236],[0.00223,0.00235],[0.00225,0.00234],[0.00223,0.00237],[0.0022,0.00241],[0.00202,0.00271],[0.00193,0.00254],[0.00189,0.00237],[0.00187,0.00224],[0.00183,0.00216],[0.00199,0.00229],[0.00203,0.0022],[0.00203,0.00225],[0.002,0.00229],[0.00206,0.0023],[0.0021,0.00223],[0.00209,0.00228],[0.00207,0.00232],[0.00213,0.00234],[0.00217,0.00227],[0.00215,0.00231],[0.00212,0.00236],[0.00219,0.00238],[0.00222,0.00233],[0.0022,0.00237],[0.00218,0.00241]]]
final =[]
for frame in seq:
    upper = frame[:7] 
    
    left_wrist = frame[7]
    left_thumb_cmc = frame[8]
    left_thumb_tip = frame[11]
    left_index_mcp = frame[12]
    left_index_tip = frame[15]
    left_middle_mcp = frame[16]
    left_middle_tip = frame[19]
    left_ring_mcp = frame[20]
    left_ring_tip = frame[23]
    left_pinky_mcp = frame[24]
    left_pinky_tip = frame[27]
    
    right_wrist = frame[28]
    right_thumb_mcp = frame[29]
    right_thumb_tip = frame[32]
    right_index_mcp = frame[33]
    right_index_tip = frame[36]
    right_middle_mcp = frame[37]
    right_middle_tip = frame[40]
    right_ring_mcp = frame[41]
    right_ring_tip = frame[44]
    right_pinky_mcp = frame[45]
    right_pinky_tip = frame[48]
    
    frame_landmarks = np.concatenate([
        upper, 
        [left_wrist], [left_thumb_cmc], [left_thumb_tip], 
        [left_index_mcp], [left_index_tip], [left_middle_mcp], [left_middle_tip], 
        [left_ring_mcp], [left_ring_tip], [left_pinky_mcp], [left_pinky_tip],
        [right_wrist], [right_thumb_mcp], [right_thumb_tip], 
        [right_index_mcp], [right_index_tip], [right_middle_mcp], [right_middle_tip], 
        [right_ring_mcp], [right_ring_tip], [right_pinky_mcp], [right_pinky_tip]
    ])
    
    # Extract only x,y coordinates (drop z)
    frame_landmarks_2d = frame_landmarks[:, :2]
    
    final.append(frame_landmarks_2d)

show_output.save_generated_sequence(final, CVAE_OUTPUT_FRAMES, CVAE_OUTPUT_VIDEO)
