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

## Day 2 — LiDAR to Camera Projection

### Key Concepts

- **LiDAR coordinate frame (Velodyne)**
  - x: forward
  - y: left
  - z: up
- `lidar[:, 2]` is height relative to LiDAR sensor origin

### Projection to Camera

1. Convert LiDAR points to homogeneous coordinates: `[x, y, z, 1]`
2. Transform to camera frame.
3. Project to image plane.
4. Mask points outside image & filter points behind camera.

### Insights
- Homogeneous coordinates allow rotation + translation in one step
- Projection order matters: LiDAR -> Camera -> Rectified -> Image
- Masking & filtering ensures physically meaningful visualization
- Ground-relative height calculation adds semantic understanding
