import torch
from dataset.kitti_dataset import KittiTrackingDataset
from bev.bev_config import BEVConfig

def main():
    cfg = BEVConfig()

    ds = KittiTrackingDataset(
        root_dir="data/KITTI/tracking/training",
        sequence="0000",
        classes=("Car",),
        frame_stride=10,
        cfg=cfg,
    )

    print("len(dataset):", len(ds))
    sample = ds[0]

    # Basic key checks
    expected_keys = {"bev", "heatmap", "size", "yaw"}
    assert expected_keys.issubset(sample.keys()), f"Missing keys: {expected_keys - set(sample.keys())}"

    bev = sample["bev"]
    heatmap = sample["heatmap"]
    size = sample["size"]
    yaw = sample["yaw"]

    # Shape checks (bev is C,H,W)
    assert bev.ndim == 3 and bev.shape[0] == 3, f"bev shape expected (3,H,W), got {bev.shape}"
    assert heatmap.ndim == 3 and heatmap.shape[0] == 1, f"heatmap expected (1,H',W'), got {heatmap.shape}"
    assert size.ndim == 3 and size.shape[0] == 2, f"size expected (2,H',W'), got {size.shape}"
    assert yaw.ndim == 3 and yaw.shape[0] == 1, f"yaw expected (1,H',W'), got {yaw.shape}"

    # Dtype checks
    assert bev.dtype == torch.float32, f"bev dtype expected float32, got {bev.dtype}"
    assert heatmap.dtype == torch.float32, f"heatmap dtype expected float32, got {heatmap.dtype}"

    print("BEV:", bev.shape, bev.dtype)
    print("heatmap", heatmap.shape, heatmap.dtype, float(heatmap.max()))
    print("size", size.shape, size.dtype, float(size.max()))
    print("yaw", yaw.shape, yaw.dtype, float(yaw.max()))
    print("basic key checks: OK")

if __name__ == "__main__":
    main()
