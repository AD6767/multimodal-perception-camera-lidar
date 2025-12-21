# Multi-Modal Perception: Camera + LiDAR (KITTI)

This project implements a foundational **multi-modal perception pipeline**
using **camera and LiDAR data**, inspired by modern autonomous driving systems.

---

## Features
- KITTI dataset support (camera + Velodyne LiDAR)
- LiDAR → camera projection using calibration matrices
- Ground-relative LiDAR height visualization
- Bird's Eye View (BEV) construction
- Multi-channel BEV tensor:
  - Height
  - Density
  - Intensity
- Side-by-side Camera + BEV visualization

---

## Key Concepts Demonstrated
- Sensor calibration & coordinate frames
- Homogeneous transformations
- Camera projection geometry
- Ground-relative height estimation
- Metric BEV representation
- Multi-modal sensor alignment

---

## BEV Representation
The BEV encodes LiDAR data into a grid-based tensor suitable for
detection and tracking models.

**Channels**
- **Height**: Max ground-normalized height per cell
- **Density**: Log-normalized point count
- **Intensity**: Mean LiDAR reflectance

**Coordinate Convention**
- X: forward
- Y: lateral
- Z: up
- Ego vehicle located at bottom-center of BEV

---

## BEV Feature Visualization

Below is an example of the BEV feature channels constructed from LiDAR data.

- **Height** highlights object structure
- **Density** captures occupancy and vertical surfaces
- **Intensity** reflects material and surface properties

![BEV Height, Density, Intensity](assets/bev_height_density_intensity.png)

---

## Visualizations
- Camera view with LiDAR height overlay
- BEV height, density, and intensity maps
- Side-by-side Camera + BEV comparison

---

## Tech Stack
- Python
- NumPy
- OpenCV
- Matplotlib

---

## Dataset
- KITTI (Camera + Velodyne LiDAR)

---

## Project Structure

```
data/ # Dataset loaders
utils/ # Calibration parsing and geometry utilities
visualization/ # Camera & BEV visualizations
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


