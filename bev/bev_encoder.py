import numpy as np


def make_bev_tensor(lidar: np.ndarray, res: float = 0.1, x_range=(0,50), y_range=(-25,25), height_range=(0.0, 3.0), max_points: int = 64):
    """
    Build a BEV tensor from LiDAR points.
    Args:
        lidar: Array of shape (N, 4) containing LiDAR points [x, y, z, intensity].
        res: Resolution of each BEV grid cell in meters. (meters per pixel)        
        x_range: Forward distance range in meters (xmin, xmax).
        y_range: Lateral distance range in meters (ymin, ymax).
        height_range: Height range for normalization (zmin, zmax).
        max_points (int): Maximum number of points per grid cell for density normalization.
    Returns:
        bev_tensor: (C, H, W) float32 where C=3 [height, density, intensity]
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    # filter ROI in metric space (use original z for mask)
    mask = (lidar[:, 0] >= x_min) & (lidar[:, 0] <= x_max) & (lidar[:, 1] >= y_min) & (lidar[:, 1] <= y_max)
    points = lidar[mask]
    if points.shape[0] == 0:
        # Return empty BEV tensor if no points in ROI
        H = int((x_max - x_min) / res)
        W = int((y_max - y_min) / res)
        return np.zeros((3, H, W), dtype=np.float32)
    
    # ground normalize z
    z = points[:, 2]
    ground_z = np.percentile(z, 5)
    z_ground = np.clip(z - ground_z, height_range[0], height_range[1])

    # to pixel indices
    px = ((points[:, 0] - x_min) / res).astype(np.int32)
    py = ((points[:, 1] - y_min) / res).astype(np.int32)

    H = int ((x_max - x_min) / res)
    W = int ((y_max - y_min) / res)

    # bounds safety
    valid = (px >= 0) & (px < H) & (py >= 0) & (py < W)
    px = px[valid]
    py = py[valid]
    z_ground = z_ground[valid]
    points = points[valid]

    # Initialize BEV channels
    height_map = np.zeros((H, W), dtype=np.float32)
    density_map = np.zeros((H, W), dtype=np.float32)
    intensity_map = np.zeros((H, W), dtype=np.float32)
    intensity_sum = np.zeros((H, W), dtype=np.float32)
    intensity_count = np.zeros((H, W), dtype=np.float32)

    # Fill maps
    for x, y, z, intensity in zip(px, py, z_ground, points[:, 3]):
        # height map: max height
        if z > height_map[x, y]:
            height_map[x, y] = z
        # density map: count points
        density_map[x, y] += 1
        # intensity map: sum and count for average
        intensity_sum[x, y] += float(intensity)
        intensity_count[x, y] += 1

    # Normalize density map
    density_map = np.clip(np.log1p(density_map) / np.log1p(max_points), 0.0, 1.0)
    
    mask_i = intensity_count > 0
    intensity_map[mask_i] = intensity_sum[mask_i] / intensity_count[mask_i]

    # output (C, H, W) for PyTorch
    bev_tensor = np.stack([height_map, density_map, intensity_map], axis=0).astype(np.float32)
    return bev_tensor

