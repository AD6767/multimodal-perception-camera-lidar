# Notes -- Multi-Modal Perception (Camera + LiDAR)

These notes capture key concepts and insights learned while building a multi-modal perception pipeline.

## Part 1 — Sensor & Geometry Fundamentals

### LiDAR
- Stored as binary float32: (x, y, z, reflectance)
- x: forward, y: left, z: up
- Typical frame contains ~100k points
- BEV (X-Y) is the standard view for detection & tracking

### Camera
- Perspective projection
- Rectified images used for geometry alignment

### Calibration
- Extrinsics: LiDAR -> camera transformation
- Rectification aligns camera to a common stereo frame
- Projection matrix maps 3D camera coordinates -> 2D pixels
- Calibration parsing is dataset-specific

### Key Insight
Always verify raw sensor data and geometry before applying ML models.
Small calibration errors propagate downstream into fusion and tracking.

