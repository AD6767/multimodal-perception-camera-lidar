import numpy as np
import matplotlib.pyplot as plt
import cv2


def project_lidar_to_image(lidar, P2, R0, Tr, img_shape=None):
    """
    Project LiDAR points into the camera image plane.

    Args:
        lidar: np.array (N,4) LiDAR points [x,y,z,intensity]
        P2: np.array (3,4) Camera projection matrix
        R0: np.array (3,3) Rectification matrix
        Tr: np.array (3,4) LiDAR->Camera transform
        img_shape: tuple (H,W) for filtering points outside image

    Returns:
        u, v: pixel coordinates (N,)
        mask: boolean array of points inside image (if img_shape given)
    """
    N = lidar.shape[0]
    # 1. Convert LiDAR to homogeneous coordinates
    lidar_h = np.hstack((lidar[:, :3], np.ones((N, 1))))  # (N,4)

    # 2. Transform to camera frame
    X_cam = (Tr @ lidar_h.T)    # (3,N)
    X_rect = (R0 @ X_cam)       # (3,N) rectified camera frame

    # 3. Project to image plane
    X_img = P2 @ np.vstack((X_rect, np.ones((1,N))))     # (3,N)
    u = X_img[0,:] / X_img[2,:]
    v = X_img[1,:] / X_img[2,:]

    # 4. filter points inside image
    mask = np.ones(N, dtype=bool)
    if img_shape is not None:
        H, W = img_shape
        mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (X_rect[2,:] > 0) # avoids plotting points outside camera FOV
    
    return u, v, mask

def overlay_lidar_on_image(img_path, lidar, P2, R0, Tr):
    """Overlay LiDAR points onto camera image"""
    img = cv2.imread(img_path)  # BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting

    u, v, mask = project_lidar_to_image(lidar, P2, R0, Tr, img.shape[:2])

    z = lidar[:, 2]
    ground_z = np.percentile(z, 5)
    z_ground = z - ground_z # frame-adaptive ground height
    z_vis = np.clip(z_ground, 0.0, 3.0) # Clip extreme values to improve contrast

    plt.figure(figsize=(12, 5))
    plt.imshow(img)
    plt.scatter(u[mask], v[mask], s=0.5, c=z_vis[mask], cmap="viridis")  # color by height
    plt.axis("off")
    plt.title("LiDAR → Camera Projection (Height-normalized)")
    plt.colorbar(label="Height above ground (m)")
    plt.show()

