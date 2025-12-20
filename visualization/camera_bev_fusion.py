import matplotlib.pyplot as plt
import cv2
import numpy as np

from visualization.bev import lidar_to_bev_height
from visualization.project_lidar_to_image import project_lidar_to_image


def visualize_camera_and_bev(
    img_path,
    lidar,
    P2,
    R0,
    Tr,
    bev_res=0.1,
    x_range=(0, 50),
    y_range=(-25, 25),
    height_range=(0.0, 3.0)
):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Height normalization
    z = lidar[:, 2]
    ground_z = np.percentile(z, 5)
    z_ground = z - ground_z
    z_vis = np.clip(z_ground, height_range[0], height_range[1])

    # Camera projection
    u, v, mask = project_lidar_to_image(lidar, P2, R0, Tr, img.shape[:2])

    # BEV
    bev_map = lidar_to_bev_height(lidar, res=bev_res, x_range=x_range, y_range=y_range, height_range=height_range)

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Camera view
    axs[0].imshow(img)
    sc = axs[0].scatter(u[mask], v[mask], s=0.5, c=z_vis[mask], cmap="viridis", vmin=height_range[0], vmax=height_range[1])
    axs[0].set_title("Camera View (LiDAR Height Overlay)")
    axs[0].axis("off")
    # BEV view
    axs[1].imshow(bev_map, cmap="viridis", origin="lower", vmin=height_range[0], vmax=height_range[1])
    axs[1].set_title("LiDAR Bird's Eye View (Height)")
    axs[1].set_xlabel("Lateral (Y)")
    axs[1].set_ylabel("Forward (X)")
    fig.colorbar(sc, ax=axs[0], label="Height above ground (m)")

    print("z_ground stats:", np.min(z_ground), np.percentile(z_ground, [25, 50, 75]), np.max(z_ground))
    
    plt.tight_layout()
    plt.show()
