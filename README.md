# Classroom Proctor AI (YOLOv8 Webcam)

This is a simple real-time webcam demo that uses a YOLOv8 model to detect objects and flag phones on screen. When a cell phone is detected with >50% confidence, the overlay turns red and a warning message appears.

## What it does
- Opens your default webcam.
- Runs YOLOv8 on each frame.
- Draws bounding boxes with labels and confidence.
- Highlights a phone detection with a red alert overlay.
- Press `q` to quit.

## Requirements
- Python 3.8+
- A working webcam
- Packages: `opencv-python`, `ultralytics`

## Setup
1. Create and activate a virtual environment (optional).
2. Install dependencies:

```bash
pip install opencv-python ultralytics
```

## Run
```bash
python main.py
```

On first run, the model file `yolov8n.pt` will be loaded from the local directory (and downloaded automatically if missing).

## Notes
- If the webcam fails to open, try changing `cv2.VideoCapture(0)` to `1` or `2` in `main.py`.
- Detection confidence is filtered at 0.5 in the script.

## Project files
- `main.py` — main application logic
- `yolov8n.pt` — YOLOv8n model weights
