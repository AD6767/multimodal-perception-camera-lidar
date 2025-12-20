# Multi-Modal Perception: Camera + LiDAR (KITTI)

This repository explores multi-modal 3D perception using **camera and LiDAR data**,
focusing on detection, segmentation, and tracking for autonomous driving.

The project is built incrementally, with each stage emphasizing **sensor geometry,
calibration, and fusion fundamentals** before introducing learning-based models.

---

## Dataset
- **KITTI Tracking Dataset**
- Sensors:
  - RGB camera (rectified)
  - Velodyne LiDAR (3D point clouds)

Dataset is expected under: `dataset/KITTI/`

## Part 1 -- Dataset & Sensor Foundations
- Load raw camera images and LiDAR point clouds
- Parse and validate calibration files
- Understand sensor coordinate frames
- Visualize raw sensor data
This establishes a reliable foundation for camera-LiDAR fusion.

## Part 2 -- LiDAR to Camera Projection
- Projected LiDAR points into camera image using KITTI calibration
- Filtered points behind camera and outside image bounds
- Visualized points colored by **ground-relative height**

## Part 3 -- Bird's Eye View (BEV)
- Converted LiDAR point clouds into BEV representation
- Applied ROI filtering and metric discretization
- Visualized max-height BEV with Cartesian orientation
- BEV used as foundation for detection and tracking

---

## Project Structure
```
data/ # Dataset loaders and sanity checks
utils/ # Calibration parsing and geometry utilities
visualization/ # Sensor visualizations (BEV, images)
models/ # Models (to be added)
notes/notes.md  # Learning & reference notes
dataset/ # Local dataset (ignored in git)
```

---

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run
```bash
python -m test.test
```


