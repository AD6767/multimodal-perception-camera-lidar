import numpy as np


def world_to_bev(x, y, config):
    """
    Convert metric coordinates to BEV pixel indices
    """
    px = int((x - config.X_MIN) / config.RESOLUTION)
    py = int((y - config.Y_MIN) / config.RESOLUTION)
    return px, py

def kitti_object_to_bev_box(obj) -> np.array:
    """
    Convert a KITTI object to BEV box format.
    BEV box: (x, y, w, l, yaw)

    Assumption:
    - Using LiDAR-like BEV frame
    - x forward, y left
    """
    x_bev = obj.z
    y_bev = -obj.x
    w = obj.w
    l = obj.l
    yaw = obj.yaw
    return np.array([x_bev, y_bev, w, l, yaw])

