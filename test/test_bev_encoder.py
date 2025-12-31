from dataset.kitti_loader import load_lidar
from bev.bev_encoder import make_bev_tensor

lidar = load_lidar("data/KITTI/tracking/training/velodyne/0000/000000.bin")
bev = make_bev_tensor(lidar)

print("lidar:", lidar.shape)
print("bev:", bev.shape)  # expect (3, H, W)
print("bev dtype:", bev.dtype)
