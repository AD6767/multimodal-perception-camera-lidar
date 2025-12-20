# Multi-Modal Perception: Camera + LiDAR (KITTI)

This project implements a foundational multi-modal perception pipeline using
camera and LiDAR data, inspired by autonomous driving systems.

---

## Features
- KITTI dataset support
- LiDAR -> Camera projection using calibration
- Height-normalized LiDAR visualization
- Bird’s Eye View (BEV) construction
- Side-by-side Camera + BEV visualization

## Key Concepts Demonstrated
- Sensor calibration and coordinate frames
- Homogeneous transformations
- Ground-relative height estimation
- Metric BEV representation
- Multi-modal consistency

## Visualizations
- Camera view with LiDAR height overlay
- BEV using max-height pooling per cell

## Tech Stack
- Python
- NumPy
- OpenCV
- Matplotlib

## Dataset
- KITTI (Camera + Velodyne LiDAR)

## Structure
```
data/ # Dataset loaders
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


