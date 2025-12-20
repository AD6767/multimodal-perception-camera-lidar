import numpy as np
import matplotlib.pyplot as plt


def lidar_to_bev_height(lidar, res=0.1, x_range=(0,50), y_range=(-25,25), height_range=(0.0, 3.0)):
    """
    Create height-based BEV map from LiDAR. Height is normalized relative to estimated ground plane.

    Args:
        lidar: np.array (N,4) [x,y,z,intensity]
        res: resolution in meters per pixel
        x_range: forward distance range (meters)
        y_range: lateral range (meters)
        clip_height: (min,max) height clipping for visualization

    Returns:
        bev_map: 2D numpy array
    """
    # Height normalization
    z = lidar[:, 2]
    ground_z = np.percentile(z, 5)
    z_ground = z - ground_z

    # Filter points within ROI
    mask = (
        (lidar[:,0] >= x_range[0]) & (lidar[:,0] <= x_range[1]) &
        (lidar[:,1] >= y_range[0]) & (lidar[:,1] <= y_range[1]) &
        (lidar[:,2] >= height_range[0]) & (lidar[:,2] <= height_range[1])
    )
    points = lidar[mask]
    z_ground = z_ground[mask]

    # Convert to pixel coordinates
    x_img = ((points[:, 0] - x_range[0]) / res).astype(np.int32)
    y_img = ((points[:, 1] - y_range[0]) / res).astype(np.int32)

    H = int((x_range[1] - x_range[0]) / res)
    W = int((y_range[1] - y_range[0]) / res)
    bev_map = np.zeros((H, W))

    # Fill BEV map with max height (max-height pooling)
    for xi, yi, zi in zip(x_img, y_img, z_ground):
        bev_map[xi][yi] = max(bev_map[xi][yi], zi)
    
    return bev_map

def visualize_bev(bev_map):
    plt.figure(figsize=(8,5))
    plt.imshow(bev_map, cmap='viridis', origin='lower')
    plt.title("LiDAR Bird's Eye View (Height-normalized)")
    plt.xlabel("Y (lateral)")
    plt.ylabel("X (forward)")
    plt.colorbar(label="Height above ground (m)")
    plt.show()