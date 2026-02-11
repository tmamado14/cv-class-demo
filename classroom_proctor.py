import cv2
from ultralytics import YOLO

# 1. Load the YOLO model
# The first run will download 'yolov8n.pt' (approx 6MB) automatically.
# YOLO = You Only Look Once, an object detection model that finds items in images.
print("Loading AI Model...")
model = YOLO('yolov8n.pt')

# 2. Open the Webcam (0 is usually the default camera)
# If you have multiple cameras, try 1 or 2.
cap = cv2.VideoCapture(0)

# Safety Check: always confirm the camera opened successfully
if not cap.isOpened():
    print("Yun lang! Could not open the webcam.")
    print("Try changing cv2.VideoCapture(0) to 1 or 2.")
    exit()

print("Camera started! Press 'q' to quit.")

while True:
    # Read a frame from the camera
    # success is True if a frame was captured, frame is the image (numpy array)
    success, frame = cap.read()
    
    if not success:
        print("Failed to read frame.")
        break

    # 3. Run the AI on the frame
    # verbose=False stops it from spamming the terminal with text
    # results[0] is the first image's detection result
    results = model(frame, verbose=False)

    cheating_detected = False

    # 4. Process Detections
    # results[0].boxes contains the data for all detected objects
    for box in results[0].boxes:
        # Each box has: class id, class name, and confidence score
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        # Filter: Only trust detections with > 50% confidence
        # We also check the class is "cell phone" for cheating detection
        if confidence > 0.5 and class_name == 'cell phone':
            # Get coordinates for the box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cheating_detected = True
            color = (0, 0, 255) # Red (BGR format)
            label = "STOP! NO PHONES!"
            
            # Add extra warning text on screen
            cv2.putText(frame, "PHONE DETECTED!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Draw the box and text
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the output window with annotations
    cv2.imshow('Classroom Proctor AI', frame)

    # Press 'q' to quit the loop
    # cv2.waitKey(1) waits 1 ms for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup: release camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("Session ended!")
