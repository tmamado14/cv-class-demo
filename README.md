# CV Demo (YOLOv8 Webcam)

This repo contains two webcam demos using YOLOv8:
1) `classroom_proctor.py` runs a classroom proctor that flags phone use.
2) `gym_bro.py` runs a pose-based rep counter for bicep curls.

## What it does
- Opens your default webcam.
- Runs YOLOv8 on each frame.
- Draws overlays in real time.
- Press `q` to quit.

## Demos
### Classroom Proctor AI (`classroom_proctor.py`)
- Detects objects with YOLOv8n.
- If a cell phone is detected with >50% confidence, a red warning appears.
- Shows bounding boxes and warning text.

### AI Gym Bro (`gym_bro.py`)
- Uses YOLOv8n-pose to detect keypoints.
- Tracks left shoulder, elbow, and wrist.
- Counts reps when arm angle moves from down (>160 degrees) to up (<30 degrees).
- Shows a rep counter and stage (up/down).

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
### Proctor
```bash
python classroom_proctor.py
```

### Gym Bro
```bash
python gym_bro.py
```

On first run, model files will be loaded from the local directory (and downloaded automatically if missing).

## Notes
- If the webcam fails to open, try changing `cv2.VideoCapture(0)` to `1` or `2` in the script you are running.
- Detection confidence for phones is filtered at 0.5 in `classroom_proctor.py`.

## Project files
- `classroom_proctor.py` - classroom proctor logic
- `gym_bro.py` - pose-based rep counter
- `yolov8n.pt` - YOLOv8n model weights
- `yolov8n-pose.pt` - YOLOv8n pose model weights
