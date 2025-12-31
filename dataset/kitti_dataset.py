import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data import ConcatDataset

from dataset.kitti_labels import parse_kitti_label_file
from dataset.kitti_loader import load_lidar
from bev.bev_encoder import make_bev_tensor
from bev.bev_config import BEVConfig
from bev.bev_utils import kitti_object_to_bev_box
from bev.bev_targets import create_bev_targets


class KittiTrackingDataset(Dataset):
    def __init__(self, root_dir: str, sequence: str = "0000", classes=("Car",), frame_stride: int = 1, cfg: BEVConfig = None):
        """
        root_dir: Path to the KITTI tracking dataset directory = "data/KITTI/tracking/training"
        sequence: eg: "0000"
        """
        self.root_dir = root_dir
        self.sequence = sequence
        self.classes = set(classes)
        self.cfg = cfg if cfg is not None else BEVConfig()

        self.velo_dir = os.path.join(root_dir, "velodyne", sequence)
        self.label_path = os.path.join(root_dir, "label_02", f"{sequence}.txt")
        
        # Parse all labels once, group by frame
        objects = parse_kitti_label_file(self.label_path)
        self.labels_by_frame = {}
        for obj in objects:
            if obj.cls not in self.classes:
                continue
            self.labels_by_frame.setdefault(obj.frame, []).append(obj)
        
        labeled_frames = set(self.labels_by_frame.keys())
        # only keep frames that have a .bin on disk 
        velo_path = Path(self.velo_dir)
        if not velo_path.exists():
            raise FileNotFoundError(f"Missing velodyne folder: {self.velo_dir}")
        
        available_frames = set()
        for p in velo_path.glob("*.bin"):
            # files are like 000179.bin -> 179
            try:
                available_frames.add(int(p.stem))
            except ValueError:
                pass
        valid_frames = sorted(labeled_frames & available_frames)
        # apply frame stride after intersection
        self.frames = valid_frames[::frame_stride]
        # # Build an index of frames that exist on disk AND have labels
        # all_frames = sorted(self.labels_by_frame.keys())
        # self.frames = all_frames[::frame_stride] # apply frame stride

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx: int):
        frame_id = self.frames[idx]
        bin_path = os.path.join(self.velo_dir, f"{frame_id:06d}.bin")
        points = load_lidar(bin_path)  # (N,4)

        # 1. BEV input: expected (C,H,W) float32 numpy
        bev_tensor = make_bev_tensor(points, self.cfg.RESOLUTION, (self.cfg.X_MIN, self.cfg.X_MAX), (self.cfg.Y_MIN, self.cfg.Y_MAX))
        bev_tensor = torch.from_numpy(bev_tensor)  # to torch tensor
        # 2. Labels -> BEV boxes
        objects = self.labels_by_frame.get(frame_id, [])
        bev_boxes = []
        for obj in objects:
            bev_box = kitti_object_to_bev_box(obj)  # (x,y,w,l,yaw)
            bev_boxes.append(bev_box)
        # 3. Create BEV targets
        bev_targets_np = create_bev_targets(bev_boxes, self.cfg)
        bev_targets = {
            k: torch.from_numpy(v).float()
            for k, v in bev_targets_np.items()
        }
        return {
            "bev": bev_tensor,          # (3,H,W) float32 tensor
            "heatmap": bev_targets["heatmap"],  # (1,H',W') float32 tensor
            "size": bev_targets["size"],    # (2,H',W') float32 tensor
            "yaw": bev_targets["yaw"],    # (1,H',W') float32 tensor
        }
    
def list_kitti_tracking_sequences(root_dir: str):
    """
    Returns list like ['0000', '0001', ...] based on velodyne subfolders.
    """
    velodyne_dir = Path(root_dir) / "velodyne"
    if not velodyne_dir.exists():
        raise FileNotFoundError(f"Expected velodyne/ under {root_dir}")

    seqs = sorted([p.name for p in velodyne_dir.iterdir() if p.is_dir()])
    return seqs

def make_kitti_tracking_dataset_all_sequences(root_dir: str, classes=("Car",), frame_stride: int = 1, cfg=None, sequences=None):
    """
    Build a dataset over multiple sequences by concatenating per-sequence datasets.

    sequences: None -> auto-detect all sequences in root_dir/velodyne/
            or a list like ["0000", "0001"]
    """
    if sequences is None:
        sequences = list_kitti_tracking_sequences(root_dir)

    datasets = []
    for seq in sequences:
        ds = KittiTrackingDataset(root_dir=root_dir, sequence=seq, classes=classes, frame_stride=frame_stride, cfg=cfg)
        if len(ds) > 0:
            datasets.append(ds)
        else:
            print(f"[warn] skipping empty sequence {seq}")

    if len(datasets) == 0:
        raise RuntimeError("No valid sequences found (no frames with both labels and velodyne bins).")
    # If only one sequence, return the dataset directly.
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
