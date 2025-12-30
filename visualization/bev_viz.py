import numpy as np
import matplotlib.pyplot as plt


def draw_bev_boxes(ax, bev_boxes, edge_color='r'):
    for bbox in bev_boxes:
        x, y, w, l, yaw = bbox
        corners = np.array([
            [-l/2, -w/2],
            [-l/2, w/2],
            [l/2, w/2],
            [l/2, -w/2]
        ])
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        """Local box (origin)
                |
                |  rotate by yaw
                v
        Rotated box (still at origin)
                |
                |  translate by (x, y)
                v
        Final BEV box in world frame"""
        rotated_corners = corners @ R.T # corners.shape == (4, 2)
        rotated_corners[:, 0] += x # translate x
        rotated_corners[:, 1] += y # translate y
        poly = np.vstack([rotated_corners, rotated_corners[0]]) # poly.shape == (5, 2) to close the box while plotting
        ax.plot(poly[:, 0], poly[:, 1], color=edge_color) # plot point 1 to point 2, point 2 to point 3, ..., point 4 to point 1

def visualize_bev(bev_boxes, title='BEV Ground Truth'):
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_bev_boxes(ax, bev_boxes)
    ax.set_aspect("equal")
    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Y (left)")
    ax.set_title(title)
    ax.grid(True)
    plt.show()

def visualize_heatmap(heatmap, title="BEV Heatmap"):
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap[0], cmap="hot")
    plt.title(title)
    plt.colorbar()
    plt.show()
