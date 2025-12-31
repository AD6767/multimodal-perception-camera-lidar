import torch
from torch import optim

from models.bev_detector import BEVDetector
from training.losses import DetectorLoss
from training.train import train_one_epoch, make_dataloader

from dataset.kitti_dataset import KittiTrackingDataset
from bev.bev_config import BEVConfig

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    config = BEVConfig()
    dataset = KittiTrackingDataset(root_dir="data/KITTI/tracking/training", sequence="0000",cfg=config)
    loader = make_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0, device=device)

    model = BEVDetector(in_channels=3).to(device)
    loss_fn = DetectorLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_one_epoch(model, loss_fn, optimizer, loader, device=device, log_every=5, grad_clip=5.0)

if __name__ == "__main__":
    main()
