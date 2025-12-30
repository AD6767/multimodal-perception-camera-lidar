import cv2
import numpy as np

def load_image(img_path):
    """Load RGB image"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB

def load_lidar(bin_path):
    """Load LiDAR point cloud (x, y, z, reflectance)"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points # shape = (100000, 4)
