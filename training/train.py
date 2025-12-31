import os
import torch
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from bev.bev_config import BEVConfig
from models.bev_detector import BEVDetector
from training.losses import DetectorLoss
from torch.utils.data import ConcatDataset
from dataset.kitti_dataset import make_kitti_tracking_dataset_all_sequences


def make_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0, device="cpu", pin_memory=False, drop_last=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

def make_train_val_loaders(dataset, batch_size=4, num_workers=0, val_split=0.2, seed=42, pin_memory=False):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)
    train_loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = make_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    return train_loader, val_loader

def list_kitti_tracking_sequences(root_dir: str):
    """
    KITTI tracking training labels typically live at: root_dir/label_02/0000.txt, 0001.txt, ...
    treat each *.txt as one sequence id.
    """
    label_dir = os.path.join(root_dir, "label_02")
    paths = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    seqs = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return seqs

def build_tracking_dataset_all_sequences(root_dir: str, cfg, classes=("Car",), frame_stride: int = 1):
    """
    Builds a ConcatDataset over all sequences in label_02.
    Uses your existing KittiTrackingDataset(sequence=...).
    """
    from dataset.kitti_dataset import KittiTrackingDataset  # keep local import to avoid circulars

    seqs = list_kitti_tracking_sequences(root_dir)
    if len(seqs) == 0:
        raise FileNotFoundError(f"No sequences found under {root_dir}/label_02/*.txt")

    datasets = [
        KittiTrackingDataset(
            root_dir=root_dir,
            sequence=seq,
            classes=classes,
            frame_stride=frame_stride,
            cfg=cfg,
        )
        for seq in seqs
    ]
    return ConcatDataset(datasets), seqs

def save_checkpoint(path, model, optimizer, epoch, val_metrics=None):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_metrics": val_metrics or {},
        },
        path,
    )

def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt

@torch.no_grad()
def evaluate(model, loss_fn, dataloader, device="cpu"):
    model.eval()
    totals = {}
    steps = 0
    for batch in dataloader:
        bev = batch["bev"].to(device)
        targets = {k: batch[k].to(device) for k in ["heatmap", "size", "yaw"]}
        outputs = model(bev)
        losses = loss_fn(outputs, targets)

        steps += 1
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().cpu())

    return {k: totals[k] / max(steps, 1) for k in totals}

def train_one_epoch(model, loss_fn, optimizer, dataloader, device="cpu", log_every=10, grad_clip=None):
    model.train()
    running = {}
    step = 0

    for batch in dataloader:
        bev = batch["bev"].to(device)  # (B,C,H,W)

        targets = {
            "heatmap": batch["heatmap"].to(device),
            "size": batch["size"].to(device),
            "yaw": batch["yaw"].to(device),
        }

        outputs = model(bev)  # dict: heatmap/size/yaw (B,*,H // stride,W // stride)

        losses = loss_fn(outputs, targets)
        total = losses["total"]

        optimizer.zero_grad(set_to_none=True)
        total.backward()

        if grad_clip is not None: # optional gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        step += 1
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v.detach().cpu())

        if step % log_every == 0:
            avg = {k: running[k] / log_every for k in running}
            running = {}
            print("[step {}] ".format(step) + " ".join(f"{k}={avg[k]:.4f}" for k in sorted(avg)))

    return model
    

def train(model, dataset, epochs=5, lr=1e-3, batch_size=4, device="cpu", val_ratio=0.1, num_workers=0, ckpt_path="checkpoints"):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = DetectorLoss().to(device)

    train_loader, val_loader = make_train_val_loaders(
        dataset, batch_size=batch_size, num_workers=num_workers, val_split=val_ratio
    )
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    print(f"train samples: {train_samples}  val samples: {val_samples}")
    print(f"steps/epoch: {train_steps} (batch_size={batch_size}, drop_last=True)")
    print(f"val steps:   {val_steps} (batch_size={batch_size}, drop_last=True)")

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        print(f"\n=== epoch {epoch}/{epochs} ===")
        train_one_epoch(model, loss_fn, optimizer, train_loader, device=device, log_every=10)
        
        val_metrics = evaluate(model, loss_fn, val_loader, device=device)
        print("[val] " + " ".join([f"{k}={val_metrics[k]:.4f}" for k in sorted(val_metrics.keys())]))
        if ckpt_path:
            last_path = os.path.join(ckpt_path, "last.pt")
            save_checkpoint(last_path, model, optimizer, epoch, val_metrics=val_metrics)
            print(f"saved checkpoint: {last_path}")
            # best checkpoint by val total loss
            val_total = float(val_metrics.get("total", 0.0))
            if val_total < best_val:
                best_val = val_total
                best_path = os.path.join(ckpt_path, "best.pt")
                save_checkpoint(best_path, model, optimizer, epoch, val_metrics=val_metrics)
                print(f"saved best checkpoint: {best_path} (val total={best_val:.4f})")

    return model

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    config = BEVConfig()
    dataset = make_kitti_tracking_dataset_all_sequences(root_dir="data/KITTI/tracking/training", cfg=config)

    model = BEVDetector(in_channels=3).to(device)
    train(model, dataset, epochs=5, batch_size=4, device=device, val_ratio=0.1, num_workers=0, ckpt_path="checkpoints")

if __name__ == "__main__":
    main()
