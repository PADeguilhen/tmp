import cv2
import numpy as np
import pandas as pd
from rtmlib import Wholebody, draw_skeleton
import os
import sys
from tqdm import tqdm

# =========================
# Interactive inputs
# =========================
input_path = sys.argv[1]
output_csv_path = sys.argv[2]
side = sys.argv[3].strip().lower()
show_window = sys.argv[4].strip().lower()

# Validation
if not os.path.isfile(input_path):
    print(f"Input video file does not exist: {input_path}")
    sys.exit(1)

if not output_csv_path.endswith(".csv"):
    print("Invalid file extension. Please enter CSV file path.")
    sys.exit(1)

if side not in ["left", "right"]:
    print("Invalid side input. Please enter 'left' or 'right'.")
    sys.exit(1)

if show_window not in ['y', 'n', 'yes', 'no']:
    print("Invalid input for window display. Please enter 'y' or 'n'.")
    sys.exit(1)

# =========================
# CSV Columns
# =========================
COLUMNS = [
    "opposite_shoulder_x", "opposite_shoulder_y",
    "shoulder_x", "shoulder_y",
    "elbow_x", "elbow_y",
    "wrist_x", "wrist_y",

    "thumb_0_x", "thumb_0_y",
    "thumb_1_x", "thumb_1_y",
    "thumb_2_x", "thumb_2_y",

    "index_0_x", "index_0_y",
    "index_1_x", "index_1_y",
    "index_2_x", "index_2_y",
    "index_3_x", "index_3_y",

    "middle_finger_0_x", "middle_finger_0_y",
    "middle_finger_1_x", "middle_finger_1_y",
    "middle_finger_2_x", "middle_finger_2_y",
    "middle_finger_3_x", "middle_finger_3_y",

    "ring_finger_0_x", "ring_finger_0_y",
    "ring_finger_1_x", "ring_finger_1_y",
    "ring_finger_2_x", "ring_finger_2_y",
    "ring_finger_3_x", "ring_finger_3_y",

    "little_finger_0_x", "little_finger_0_y",
    "little_finger_1_x", "little_finger_1_y",
    "little_finger_2_x", "little_finger_2_y",
    "little_finger_3_x", "little_finger_3_y",

    "left_eye_x", "left_eye_y",
    "right_eye_x", "right_eye_y"
]

if side == "left":
    indices = {
        "opposite_shoulder": 6,
        "shoulder": 5,
        "elbow": 7,
        "wrist": 91,
        "excess_wrist": 9,
        "hand_start": 93,
        "hand_end": 111,
        "excess_thumb": 92,
        "left_eye_start": 66,
        "left_eye_end": 70,
        "excess_left_eye": 68,
        "right_eye_start": 60,
        "right_eye_end": 64,
        "excess_right_eye": 62
    }
else:
    indices = {
        "opposite_shoulder": 5,
        "shoulder": 6,
        "elbow": 8,
        "wrist": 112,
        "excess_wrist": 10,
        "hand_start": 114,
        "hand_end": 132,
        "excess_thumb": 113,
        "left_eye_start": 66,
        "left_eye_end": 70,
        "excess_left_eye": 68,
        "right_eye_start": 60,
        "right_eye_end": 64,
        "excess_right_eye": 62
    }

limit_detection = True
indices_tab = [
    indices["opposite_shoulder"], indices["shoulder"], indices["elbow"], indices["wrist"]
]
indices_tab.extend(range(indices["hand_start"], indices["hand_end"] + 1))
indices_tab.extend(range(indices["left_eye_start"], indices["left_eye_end"] + 1))
indices_tab.extend(range(indices["right_eye_start"], indices["right_eye_end"] + 1))
indices_tab.remove(indices["excess_left_eye"])
indices_tab.remove(indices["excess_right_eye"])

# =========================
# Load RTMPose
# =========================
pose_config = '/home/leolienne/Bureau/Code/Anonymize_offline/models/rtmw-x_8xb320-270e_cocktail14-384x288.py'
pose_checkpoint = '/home/leolienne/Bureau/Code/Anonymize_offline/models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth'

device = 'cpu'
backend = 'onnxruntime'
pose_model = Wholebody(to_openpose=False,
                      mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend, device=device)

# =========================
# Video capture
# =========================
cap = cv2.VideoCapture(input_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

data_rows = []
frame_index = 0

# =========================
# Processing loop with progress bar
# =========================
print("Starting processing... Press 'q' to quit early.")
for _ in tqdm(range(total_frames), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
        break

    keypoints, scores = pose_model(frame)
    
    # Limit keypoints to selected indices
    if limit_detection and len(keypoints) > 0:
        for i, (x, y) in enumerate(keypoints[0]):
            # Hide excess keypoints to make them invisible
            if i == indices["excess_thumb"]:
                keypoints[0][i] = keypoints[0][indices["hand_start"]] # Excessive thumb keypoint
            elif i == indices["excess_wrist"]:
                keypoints[0][i] = keypoints[0][indices["wrist"]] # Excessive wrist keypoint
            elif not i in indices_tab:
                keypoints[0][i] = keypoints[0][indices["shoulder"]] # Excessive body and face keypoints
    
    if show_window:
        #  Visualization of pose keypoints  
        draw_skeleton(frame, keypoints, scores, kpt_thr=0.4)
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit early
            break

    # Extract coordinates
    if len(keypoints) > 0:
        selected = keypoints[0][indices_tab] # Select the first (and only) person detected
        coords_flat = selected.flatten() # Flatten to 1D array

        # Average the 4 iris keypoints (4 points times 2 coordinates times 2 sides) to get center of left eye and right eye
        coords_no_eyes = coords_flat[:-16]
        coords_eyes = coords_flat[-16:]

        left_eye_x = np.mean(coords_eyes[0:8:2])
        left_eye_y = np.mean(coords_eyes[1:8:2])
        right_eye_x = np.mean(coords_eyes[8:16:2])
        right_eye_y = np.mean(coords_eyes[9:16:2])

        coords_flat = np.concatenate((coords_no_eyes, [left_eye_x, left_eye_y, right_eye_x, right_eye_y]))
    else:
        coords_flat = np.full(len(COLUMNS), np.nan)

    row = {"Frame": frame_index}
    for i, col in enumerate(COLUMNS):
        row[col] = float(coords_flat[i])

    data_rows.append(row)
    frame_index += 1

cap.release()
cv2.destroyAllWindows()

# =========================
# Save CSV
# =========================
pd.DataFrame(data_rows).astype(int).to_csv(output_csv_path, index=False)
print(f"CSV saved to: {output_csv_path}")
