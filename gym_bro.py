import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load the YOLOv8 Pose Model (It detects skeletons!)
print("Loading AI Gym Bro... (Wait lang sor)")
model = YOLO('yolov8n-pose.pt') 

# Helper Function: Calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a) # Shoulder
    b = np.array(b) # Elbow
    c = np.array(c) # Wrist
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

cap = cv2.VideoCapture(0)

# Tuning knobs
CONF_THRESH = 0.30
DOWN_ANGLE = 150
UP_ANGLE = 60

# Helper: pick an arm with enough keypoint confidence
def get_arm_points(kpts_xy, kpts_conf, side):
    if side == "left":
        idxs = (5, 7, 9)  # L_Shoulder, L_Elbow, L_Wrist
    else:
        idxs = (6, 8, 10) # R_Shoulder, R_Elbow, R_Wrist

    confs = [float(kpts_conf[i]) for i in idxs]
    if min(confs) < CONF_THRESH:
        return None, None

    pts = [[float(kpts_xy[i][0]), float(kpts_xy[i][1])] for i in idxs]
    return pts, sum(confs) / 3.0

# Variables to keep track of reps
counter = 0 
stage = None # "UP" or "DOWN"
arm_label = "--"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Run YOLO Pose
    results = model(frame, verbose=False)
    
    # Start with the raw frame; we'll draw only the arm overlay
    annotated_frame = frame.copy()
    
    # Extract Keypoints
    # YOLO Pose returns keypoints for all people detected
    # We only take the first person (results[0])
    try:
        kpts = results[0].keypoints
        if kpts is None or len(kpts) == 0:
            raise ValueError("No keypoints")

        kpts_xy = kpts.xy[0]
        kpts_conf = kpts.conf[0]

        left_pts, left_conf = get_arm_points(kpts_xy, kpts_conf, "left")
        right_pts, right_conf = get_arm_points(kpts_xy, kpts_conf, "right")

        if left_pts and right_pts:
            use_left = left_conf >= right_conf
        elif left_pts:
            use_left = True
        elif right_pts:
            use_left = False
        else:
            raise ValueError("Low confidence arm keypoints")

        if use_left:
            shoulder, elbow, wrist = left_pts
        else:
            shoulder, elbow, wrist = right_pts

        # Calculate the angle
        angle = calculate_angle(shoulder, elbow, wrist)

        # Draw only the arm (no face/eyes overlay)
        s = tuple(np.array(shoulder, dtype=int))
        e = tuple(np.array(elbow, dtype=int))
        w = tuple(np.array(wrist, dtype=int))
        cv2.line(annotated_frame, s, e, (0, 255, 0), 3)
        cv2.line(annotated_frame, e, w, (0, 255, 0), 3)
        cv2.circle(annotated_frame, s, 6, (0, 255, 0), -1)
        cv2.circle(annotated_frame, e, 6, (0, 255, 0), -1)
        cv2.circle(annotated_frame, w, 6, (0, 255, 0), -1)

        # Visualizing the Angle
        cv2.putText(annotated_frame, f"{int(angle)} deg", 
                           tuple(np.multiply(elbow, [1, 1]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # --- THE REP COUNTING LOGIC ---
        if angle > DOWN_ANGLE:
            stage = "down"
        if angle < UP_ANGLE and stage == "down":
            stage = "up"
            counter += 1
            print(f"Lohpitz! One Rep! Total: {counter}")

    except Exception:
        # Taiyaaa! Sometimes it loses the arm if you move too fast
        pass
        
    # Draw the Scoreboard
    cv2.rectangle(annotated_frame, (0,0), (225,73), (245,117,16), -1)
    
    # Rep Data
    cv2.putText(annotated_frame, 'REPS', (15,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(annotated_frame, str(counter), (10,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    
    # Stage Data (Up/Down)
    cv2.putText(annotated_frame, 'STAGE', (65,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    stage_text = stage if stage else "--"
    cv2.putText(annotated_frame, stage_text, (60,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


    # Show it
    cv2.imshow('AI Gym Trainer', annotated_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
