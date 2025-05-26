# Camera Calibration & Object Detection Project

This project includes:
- 📷 **Intrinsic Calibration** using chessboard pattern
- 📐 **Extrinsic Calibration** from image poses
- 🧠 **YOLOv8 Detection** with projection of ultrasonic readings

---

## 📁 Structure

- `scripts/`:
    - `intrinsic_calibration.py`: Performs intrinsic calibration from a video.
    - `extrinsic.py`: Computes camera pose from chessboard images.
    - `detect_general.py`: Runs YOLO detection and projects ultrasonic measurements.

---

## ⚙️ Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run scripts as needed:
```bash
python scripts/intrinsic_calibration.py
python scripts/extrinsic.py
python scripts/detect_general.py
```

---

## ✅ Requirements

```text
opencv-python
numpy
ultralytics
```

---

## 📌 Notes

- Place calibration video as `2.mp4` in the root directory.
- Place chessboard images for extrinsic in `extrinsic_images/`.
- Place a test image for detection in `detect_image/camera.jpg`.
- Pretrained YOLOv8 model (`yolov8n.pt`) must be present in the root directory.
