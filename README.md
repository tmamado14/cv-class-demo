# CV Demo (YOLOv8 Webcam)

This repo contains three webcam demos using YOLOv8:
1) `classroom_proctor.py` runs a classroom proctor that flags phone use.
2) `gym_bro.py` runs a pose-based rep counter for bicep curls.
3) `invisibility_cloak.py` runs a segmentation-based invisibility cloak.

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

### Invisibility Cloak (`invisibility_cloak.py`)
- Uses YOLOv8n-seg to segment people (class 0 only).
- Captures a static background with the `b` key.
- Replaces person pixels with background pixels using bitwise masks.
- Press `q` to exit at any time.

## Requirements
- Python 3.8+
- A working webcam
- Packages: `opencv-python`, `ultralytics`

## Clone (students)
1. Install Git if you do not already have it.
2. Open a terminal and run:

```bash
git clone https://github.com/tmamado14/cv-class-demo.git
cd cv-class-demo
```

3. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
```

Windows (PowerShell):
```bash
.\venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source venv/bin/activate
```

4. Install dependencies:

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

### Invisibility Cloak
```bash
python invisibility_cloak.py
```

On first run, model files will be loaded from the local directory (and downloaded automatically if missing).

## Notes
- If the webcam fails to open, try changing `cv2.VideoCapture(0)` to `1` or `2` in the script you are running.
- Detection confidence for phones is filtered at 0.5 in `classroom_proctor.py`.

## Project files
- `classroom_proctor.py` - classroom proctor logic
- `gym_bro.py` - pose-based rep counter
- `invisibility_cloak.py` - segmentation-based invisibility cloak
- `yolov8n.pt` - YOLOv8n model weights
- `yolov8n-pose.pt` - YOLOv8n pose model weights
- `yolov8n-seg.pt` - YOLOv8n segmentation model weights
