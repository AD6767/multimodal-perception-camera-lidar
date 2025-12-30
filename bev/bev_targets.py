import numpy as np
from bev.bev_utils import world_to_bev

def create_bev_targets(bev_boxes, config):
    """
    bev_boxes: list of (x, y, w, l, yaw)
    """

    H, W = config.HEIGHT, config.WIDTH

    heatmap = np.zeros((1, H, W), dtype=np.float32)
    size = np.zeros((2, H, W), dtype=np.float32)   # w, l
    yaw = np.zeros((1, H, W), dtype=np.float32)

    for box in bev_boxes:
        x, y, w, l, theta = box

        cx, cy = world_to_bev(x, y, config)

        if not (0 <= cx < W and 0 <= cy < H):
            continue

        heatmap[0, cy, cx] = 1.0
        size[0, cy, cx] = w
        size[1, cy, cx] = l
        yaw[0, cy, cx] = theta

    return {
        "heatmap": heatmap,
        "size": size,
        "yaw": yaw,
    }
