import numpy as np


def remove_ground_simple(lidar, z_percentile=5, threshold=0.2):
    z = lidar[:, 2]
    ground_z = np.percentile(z, z_percentile)
    mask = z > ground_z + threshold
    return lidar[mask]


