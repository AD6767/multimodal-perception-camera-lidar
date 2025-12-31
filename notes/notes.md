# Notes -- Multi-Modal Perception (Camera + LiDAR)

These notes capture key concepts and insights learned while building a multi-modal perception pipeline.

## Sensor & Geometry Fundamentals

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

## LiDAR to Camera Projection

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

## Bird's Eye View (BEV)

### Why BEV is used
- Perspective images distort scale
- BEV preserves metric distances
- Tracking and motion reasoning are simpler in BEV

### Coordinate conventions
- X: forward
- Y: lateral
- Z: height
- Ego vehicle placed at bottom of BEV

### Implementation details
- ROI filtering avoids wasted computation
- Subtract min range to map to 0-index grid
- Divide by resolution (meters -> pixels)
- Max-height per cell chosen over density

### Visualization pitfalls
- Must use origin='lower'
- Camera images use origin='upper'
- Incorrect origin flips forward/backward intuition

### Ego vs World vs Object frames
| Frame        | Meaning                             |
| ------------ | ----------------------------------- |
| Ego frame    | Coordinates relative to our vehicle |
| World frame  | Global map coordinates              |
| Object frame | Coordinates relative to an object   |

In KITTI:
- LiDAR is already in ego frame. BEV is constructed in ego frame.
- Camera projection converts ego -> camera frame.

## Height Normalization in LiDAR
Raw LiDAR Z-values are relative to the sensor origin, not the ground. Directly visualizing or thresholding Z leads to misleading results.

### Problem
- Ground points often have large negative Z
- Height distributions are skewed
- Camera and BEV visualizations become inconsistent

### Solution
Estimate ground height per frame using a robust percentile:
- Ground height ~ 5th percentile of Z
- Normalize height: z_ground = z - ground_z

This produces:
- Ground ~ 0 m
- Objects > 0 m
- Consistent height semantics across modalities

### Takeaways

- LiDAR height is sensor-centric, not ground-centric
- Ground-relative height is essential for meaningful BEV
- BEV enables metric reasoning for detection and tracking
- Consistent semantics across views is critical in multi-modal systems

## BEV Feature Channels

Raw LiDAR point clouds are sparse and unordered. BEV converts them into a dense, grid-based representation suitable for CNNs.
Each BEV cell corresponds to a fixed physical area. Inside each cell, there may be 0, 1, or many LiDAR points. We must summarize them into fixed-size features.

Common BEV channels:
### 1. Height
- Stores max height above ground per cell
- Preserves vertical structure of objects
- Useful for distinguishing vehicles, curbs, and tall obstacles

### 2. Density
- Stores number of points per BEV cell
- Indicates occupancy vs free space
- Helps networks reason about object solidity (Intuition for cluster: Road -> sparse, Car -> dense, noise -> very low density)

### 3. Intensity (Reflectance)
- Stores mean LiDAR reflectance per cell
- Captures material properties. Useful for lane markings, poles, road texture

## Why BEV is Preferred for Detection

BEV offers:
- Metric consistency (no perspective distortion)
- Fixed spatial layout
- Dense grid structure

This allows:
- Standard 2D CNNs to be applied
- Efficient convolution over space
- Easier multi-object reasoning

Most modern 3D detectors operate on BEV features.

### 1. Map LiDAR points to BEV grid
Conceptually: Each point is assigned to a discrete 2D cell in the BEV map. 
If your LiDAR covers 0–50 meters forward (x) and –25 to 25 meters sideways (y), and your BEV is 256×256, then each cell represents ~ 0.2 m × 0.2 m of the real world.
Eg: A point at `(10, 0)` maps to `(128, 51)` cell in BEV map.

Inputs:
* LiDAR points: `(N, 4)` -> `[x, y, z, intensity]`
* BEV ranges: `x_range = [xmin, xmax]`, `y_range = [ymin, ymax]`
* BEV grid: `(H, W)`

Equations:
```
delta_x = (x_max - x_min) / W # size of a cell along the x-axis
delta_y = (y_max - y_min) / H # size of a cell along the y-axis
grid_x = floor((x_i - x_min) / delta_x)
grid_y = floor((y_i - y_min) / delta_y)
```
* Subtracting `x_min` & `y_min` shifts co-ordinates so the grid starts at 0.
* Diving by `delta_x` & `delta_y` converts meters to cells.
* `floor()` converts coordinates to integer grid indices.
* Clip indices to `[0, W-1]` and `[0, H-1]` to avoid errors.

### 2. Compute Density Map
Conceptually: Density is the number of LiDAR points in a BEV cell. 
* Problem: Different cells can have very different numbers of points, eg: Close objects -> many points per cell, Far objects -> few points per cell. If we just use the raw count, high-density cells (nearby objects) dominate the BEV input and low-density cells (faraway objects) are barely visible.
* Normalization rescales the density into a fixed range [0, 1] so all cells contribute meaningfully, regardless of distance.

Equation:
```
density = min(1.0, log(1 + n_points) / log(1 + N_max))
```
* `N_max` = maximum points per cell (eg: 64)
* Result: `density_map` of shape `(H, W)` with values in [0,1].

### 3. Compute Intensity Map
Conceptually: Sum intensity of points in each cell, divide by count.
* Each BEV cell may have different numbers of points. Eg: Cell A: 2 points, intensities = [0.8, 0.6], Cell B: 20 points, intensities = many values.
* If we just sum the intensities, cells with more points will automatically have larger values, which is not what we want. We want each cell to represent the typical reflectance of that location, independent of how many points fell there. So, we compute the mean. 

Equation:
```
intensity = sum(point_intensity_in_cell) / number_of_points_in_cell
```
* If a cell has no points, intensity = 0.
* Result: `intensity_map` of shape `(H, W)`.

### Summary for the BEV channels.
| Channel   | Aggregation method | Reason                                                    |
| --------- | ------------------ | --------------------------------------------------------- |
| Height    | Max                | Capture tallest point (object structure)                  |
| Density   | Count + log norm   | Show occupancy of cell                                    |
| Intensity | Mean               | Show typical reflectivity independent of number of points |

## Ground Handling in LiDAR Processing

### Simple Ground Removal (Baseline)
- Ground is estimated using a low percentile of LiDAR z-values (e.g., 5th percentile)
- Assumes road surface occupies the lowest heights in the scene
- Ground-relative height computed as:
  - `z_ground = z - ground_z`
- Heights are clipped to a fixed range (e.g., 0–3 m)

## BEV as CNN Input

- BEV tensors are treated as images by CNNs
- Each channel encodes complementary geometric information:
  - Height -> object shape
  - Density -> point distribution
  - Intensity -> surface reflectance

Compared to RGB:
- No perspective distortion
- Metric distances preserved
- Objects have consistent scale

## What BEV Detectors Predict

In BEV space, detectors typically predict:
- Object center (x, y)
- Object size (length, width)
- Orientation (yaw) (Yaw represents object rotation around the vertical Z axis, defined in X-Y plane)
- Class probability

BEV simplifies detection because:
- No depth ambiguity
- Rotation handled explicitly
- Ground plane already normalized

## Anchor-Based vs Anchor-Free Detection (BEV)

Anchor-Based (eg: PointPillars, SECOND):
- Uses predefined box templates
- Requires tuning anchor sizes
- Historically common

Anchor-Free (eg: CenterPoint):
- Predicts object centers directly
- Simpler formulation
- Dominant in modern BEV detectors

## Coordinate Conversion: KITTI 3D -> BEV
KITTI co-ordinates:
x -> right
y -> down (vertical)
z -> forward

KITTI 3D Label Fields:
h, w, l   -> object dimensions
x, y, z   -> object center in camera coordinates
rotation_y (yaw) -> rotation around vertical (Y) axis

BEV Coordinate System (Top-Down):
bev_x = camera x   (right)
bev_y = camera z   (forward)
BEV Box Definition: (center_x, center_y, width, length, yaw)

| KITTI 3D   | BEV       |
| ---------- | --------- |
| x          | center_x  |
| z          | center_y  |
| w          | width     |
| l          | length    |
| rotation_y | yaw       |
| y          | X dropped |
| h          | X dropped |

### BEV Targets
- Anchor-free, center-based target representation
- Heatmap: object center
- Size: width & length regression
- Yaw: orientation regression
- Grid resolution and BEV bounds control coverage & sparsity
- Cells outside BEV range skipped

## BEV Resolution, Stride, and Backbone
### BEV Resolution vs Backbone Stride
- BEV resolution defines how much real-world area one input pixel covers `resolution = meters per pixel`
- Backbone stride defines how many BEV pixels are merged into one output cell `real-world size per output cell = BEV resolution × backbone stride`
Example: each output feature corresponds to ~1 meter in the real world.
```
BEV resolution = 0.25 m
Backbone stride = 4
1 output cell = 4 × 0.25 m = 1.0 m × 1.0 m
```
* Why ~1 m Cells Are Good for Cars? Typical car dimensions ~ 1.8m X 4.0m. This is ideal because objects span multiple cells (not too coarse), center localization is stable.
* Backbone purpose: aggregate local geometry, increase receptive field, preserve spatial correspondence for detection

## Detection Head + Losses
- Multi-head outputs per feature cell:
  - heatmap logits: (B,1,H',W')
  - size: (B,2,H',W') for (w,l)
  - yaw: (B,1,H',W')
- Loss:
  - heatmap: BCEWithLogits (dense)
  - size + yaw: L1 at object centers only (mask from target heatmap)
- Decode:
  - sigmoid heatmap -> pick top-K peaks -> read size/yaw at those pixels -> convert to meters.

