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

def map_lidar_to_bev_grid(points, x_range=(0,50), y_range=(-25,25), bev_width=256, bev_height=256):
    """
    Convert real-world LiDAR (x, y) coordinates into discrete BEV grid indices.
    Args:
        points (np.ndarray): Array of shape (N, 4) containing LiDAR points [x, y, z, intensity].
        x_range (tuple[float, float]): Forward distance range in meters (xmin, xmax).
        y_range (tuple[float, float]): Lateral distance range in meters (ymin, ymax).
        bev_width (int): Width of the BEV grid (number of cells along x-axis).
        bev_height (int): Height of the BEV grid (number of cells along y-axis).
    Returns:
        x_indices (np.ndarray): Array of shape (N,) containing grid x indices for each point.
        y_indices (np.ndarray): Array of shape (N,) containing grid y indices for each point.
    """
    x_min, x_max = x_range[0], x_range[1]
    y_min, y_max = y_range[0], y_range[1]
    # Compute resolution (delta_x, delta_y)
    x_res = (x_max - x_min) / bev_width
    y_res = (y_max - y_min) / bev_height
    # Convert x, y to BEV indices
    x_indices = np.floor((points[:, 0] - x_min) / x_res).astype(np.int32)
    y_indices = np.floor((points[:, 1] - y_min) / y_res).astype(np.int32)
    # Clip indices to be inside grid
    x_indices = np.clip(x_indices, 0, bev_width - 1)
    y_indices = np.clip(y_indices, 0, bev_height - 1)
    
    return x_indices, y_indices

def compute_bev_height_map(points, x_indices, y_indices, bev_width=256, bev_height=256, height_range=(0.0, 3.0)):
    """
    Docstring for compute_bev_height_map
    
    :param points: Description
    :param x_indices: Description
    :param y_indices: Description
    :param bev_width: Description
    :param bev_height: Description
    """
    height_map = np.zeros((bev_width, bev_height), dtype=np.float32)
    # Height normalization
    z = points[:, 2]
    ground_z = np.percentile(z, 5)
    z_ground = z - ground_z
    z_ground = np.clip(z_ground, height_range[0], height_range[1])
    # Fill BEV map with max height
    for x, y, z in zip(x_indices, y_indices, z_ground):
        height_map[x, y] = max(height_map[x, y], z)
    return height_map

def compute_bev_density_map(x_indices, y_indices, bev_width=256, bev_height=256, max_points=64):
    """
    Compute the normalized density map for a BEV grid.
    Args:
        x_indices (np.ndarray): Array of shape (N,) containing grid x indices for each point.
        y_indices (np.ndarray): Array of shape (N,) containing grid y indices for each point.
        bev_width (int): Width of the BEV grid (number of cells along x-axis).
        bev_height (int): Height of the BEV grid (number of cells along y-axis).
        max_points (int, optional): Maximum expected points per cell for normalization. Default is 64.

    Returns:
        density_map (np.ndarray): Array of shape (bev_height, bev_width) with values in [0, 1], 
                                  representing the normalized point density per cell.
    """
    density_map = np.zeros((bev_width, bev_height), dtype=np.float32)
    # Count points in the cell
    for x, y in zip(x_indices, y_indices):
        density_map[x, y] += 1
    
    density_map = np.clip(np.log1p(density_map) / np.log1p(max_points), 0, 1)
    return density_map

def compute_bev_intensity_map(points, x_indices, y_indices, bev_width=256, bev_height=256):
    """
    Compute the mean intensity map for a BEV grid.

    Args:
        points (np.ndarray): Array of shape (N, 4) containing LiDAR points [x, y, z, intensity].
        x_indices (np.ndarray): Array of shape (N,) containing grid x indices for each point.
        y_indices (np.ndarray): Array of shape (N,) containing grid y indices for each point.
        bev_height (int): Height of the BEV grid (number of cells along y-axis).
        bev_width (int): Width of the BEV grid (number of cells along x-axis).

    Returns:
        intensity_map (np.ndarray): Array of shape (bev_height, bev_width) with values representing 
                                    the mean reflectance per cell. Cells with no points have value 0.
    """
    intensity_map = np.zeros((bev_width, bev_height), dtype=np.float32)
    count_map = np.zeros((bev_width, bev_height), dtype=np.int32)

    for idx, (x, y) in enumerate(zip(x_indices, y_indices)):
        intensity_map[x, y] += points[idx, 3] # intensity
        count_map[x, y] += 1
    
    mask = count_map > 0 # avoid division by 0
    intensity_map[mask] /= count_map[mask]
    return intensity_map

def create_bev_tensor(height_map, density_map, intensity_map):
    """
    Stack height, density, and intensity maps to create a BEV tensor.

    Args:
        height_map (np.ndarray): Array of shape (H, W) representing the max height per cell.
        density_map (np.ndarray): Array of shape (H, W) representing normalized density per cell.
        intensity_map (np.ndarray): Array of shape (H, W) representing mean intensity per cell.

    Returns:
        bev_tensor (np.ndarray): Array of shape (H, W, 3), where channels are:
                                0 - Height, 1 - Density, 2 - Intensity.
    """
    bev_tensor = np.stack([height_map, density_map, intensity_map], axis=-1)
    return bev_tensor # shape = (H, W, 3)

def visualize_height_bev(bev_map):
    plt.figure(figsize=(8,5))
    plt.imshow(bev_map, cmap='viridis', origin='lower')
    plt.title("LiDAR Bird's Eye View (Height-normalized)")
    plt.xlabel("Y (lateral)")
    plt.ylabel("X (forward)")
    plt.colorbar(label="Height above ground (m)")
    plt.show()

def visualize_bev(height_map, density_map, intensity_map):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(height_map, cmap='viridis', origin='lower')
    axs[0].set_title("Height Map (ground-relative)")
    axs[0].axis("off")

    axs[1].imshow(density_map, cmap='hot', origin='lower')
    axs[1].set_title("Density Map (log-normalized)")
    axs[1].axis("off")

    axs[2].imshow(intensity_map, cmap='gray', origin='lower')
    axs[2].set_title("Intensity Map (mean reflectance)")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
