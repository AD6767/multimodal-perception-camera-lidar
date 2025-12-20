import matplotlib.pyplot as plt
import numpy as np

def show_image(img):
    plt.imshow(img)
    plt.axis("off")
    plt.title("Camera Image")
    plt.show()

def show_lidar_topdown(lidar):
    plt.figure(figsize=(8, 6))
    z = lidar[:, 2]
    z_clipped = np.clip(z, -2.5, 2.0) # LiDAR Z has outliers (trees, noise). Clip to a reasonable range.
    plt.scatter(
        lidar[:, 0],        # X (forward)
        lidar[:, 1],        # Y (left/right)
        c=z_clipped,        # Z (height)
        s=0.5,
        cmap="viridis"
    )
    plt.colorbar(label="Height (m)")
    plt.show()
