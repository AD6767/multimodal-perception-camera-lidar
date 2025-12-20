from data.kitti_loader import load_image, load_lidar
from utils.calibration import load_calibration, get_calib_matrices
from visualization.visualize_data import show_image, show_lidar_topdown

import matplotlib.pyplot as plt

lidar = load_lidar("dataset/KITTI/training/velodyne/0000/000000.bin")
print("LiDAR shape", lidar.shape)  # (123397, 4)
# uncomment to visualize sample lidar BEV. 
# show_lidar_topdown(lidar=lidar)

calib = load_calibration("dataset/KITTI/training/calib/0000.txt")
P2, R0, Tr = get_calib_matrices(calib)
print("Calibration", "P2:", P2.shape, "R0:", R0.shape, "Tr:", Tr.shape) # P2: (3, 4) R0: (3, 3) Tr: (3, 4)

img = load_image("dataset/KITTI/tracking/training/image_02/0000/000000.png")
print("Camera L", img.shape)   # (375, 1242, 3)
# uncomment to visualize sample raw image. 
# show_image(img=img)

