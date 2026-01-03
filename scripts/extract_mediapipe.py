import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
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
        "opposite_shoulder": 12,
        "shoulder": 11,
        "elbow": 13,         
        "wrist": 15   
    }
else :
    indices = {
        "opposite_shoulder": 11,
        "shoulder": 12,
        "elbow": 14,         
        "wrist": 16
    }

finger_indices = {
    "wrist": 0,
    "thumb_0": 2,
    "thumb_1": 3,
    "thumb_2": 4,
    "index_0": 5,
    "index_1": 6,
    "index_2": 7,
    "index_3": 8,
    "middle_finger_0": 9,
    "middle_finger_1": 10,
    "middle_finger_2": 11,
    "middle_finger_3": 12,
    "ring_finger_0": 13,
    "ring_finger_1": 14,
    "ring_finger_2": 15,
    "ring_finger_3": 16,
    "little_finger_0": 17,
    "little_finger_1": 18,
    "little_finger_2": 19,
    "little_finger_3": 20
}

eyes_indices = {
    "left_eye": 473,
    "right_eye": 468
}
    
# =========================
# Load MediaPipe
# =========================
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(static_image_mode=False,
                                model_complexity=2,
                                smooth_landmarks=True,
                                enable_segmentation=False,
                                refine_face_landmarks=True)

# =========================
# Video capture
# =========================
cap = cv2.VideoCapture(input_path)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
data = []

# =========================
# Processing loop with progress bar
# =========================
print("Starting processing... Press 'q' to quit early.")
for _ in tqdm(range(total_frames), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB before processing
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # Draw landmarks on frame
    annotated_frame = frame.copy()
    
    # Draw landmarks on the frame
    mp_drawing.draw_landmarks(annotated_frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    
    if side == "left":
        mp_drawing.draw_landmarks(annotated_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    elif side == "right":
        mp_drawing.draw_landmarks(annotated_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Show annotated frame
    cv2.imshow('MediaPipe Holistic', annotated_frame)

    # Extract coordinates
    if results.pose_landmarks:
        pose = results.pose_landmarks.landmark
    else:
        pose = [None] * 33
        
    if side == "left" and results.left_hand_landmarks:
        hand = results.left_hand_landmarks.landmark
        use_hand_for_wrist = True # If hand is detected, use hand base as wrist keypoint
    elif side == "right" and results.right_hand_landmarks:
        hand = results.right_hand_landmarks.landmark
        use_hand_for_wrist = True # If hand is detected, use hand base as wrist keypoint
    else:
        hand = [None] * 21
        use_hand_for_wrist = False # If hand isn't detected, use body wrist keypoint
        
    if results.face_landmarks:
        face = results.face_landmarks.landmark
    else:
        face = [None] * 478
        
    # Create new row of data
    row = []

    # Pose (shoulders and elbow)
    for key in ["opposite_shoulder", "shoulder", "elbow"]:
        idx = indices[key]
        if pose[idx] is not None:
            row.extend([int(pose[idx].x * w), int(pose[idx].y * h)])
        else:
            row.extend([np.nan, np.nan])
    
    # Wrist : from hand if detected, else from pose/body
    if use_hand_for_wrist and hand[finger_indices["wrist"]] is not None:
        row.extend([int(hand[finger_indices["wrist"]].x * w), int(hand[finger_indices["wrist"]].y * h)])
    else:
        idx = indices["wrist"]
        if pose[idx] is not None:
            row.extend([int(pose[idx].x * w), int(pose[idx].y * h)])
        else:
            row.extend([np.nan, np.nan])

    # Hand (thumb → little finger)
    for key in finger_indices.keys():
        idx = finger_indices[key]
        
        if idx == 0:
            continue  # Skip wrist
        
        if hand[idx] is not None:
            row.extend([int(hand[idx].x * w), int(hand[idx].y * h)])
        else:
            row.extend([np.nan, np.nan])

    # Eyes
    for key in eyes_indices.keys():
        idx = eyes_indices[key]
        if face[idx] is not None:
            row.extend([int(face[idx].x * w), int(face[idx].y * h)])
        else:
            row.extend([np.nan, np.nan])
    
    data.append(row)
        

    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit early
        break

cap.release()
cv2.destroyAllWindows()

# =========================
# Save CSV
# =========================
df = pd.DataFrame(data, columns=COLUMNS)
df.to_csv(output_csv_path, index=False)
print(f"CSV saved to: {output_csv_path}")
