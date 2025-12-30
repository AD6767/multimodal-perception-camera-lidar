import torch
from training.losses import DetectionLoss

def main():
    B, H, W = 1, 100, 100

    preds = {
        "heatmap": torch.randn(B, 1, H, W),
        "size": torch.randn(B, 2, H, W),
        "yaw": torch.randn(B, 1, H, W),
    }

    targets = {
        "heatmap": torch.zeros(B, 1, H, W),
        "size": torch.zeros(B, 2, H, W),
        "yaw": torch.zeros(B, 1, H, W),
    }

    targets["heatmap"][0, 0, 50, 50] = 1.0

    loss_fn = DetectionLoss()
    losses = loss_fn(preds, targets)

    for k, v in losses.items():
        print(k, v.item())
    """
    loss_total 2.153339385986328
    loss_heatmap 0.35161200165748596
    loss_size 1.472015142440796
    loss_yaw 0.32971227169036865
    """

if __name__ == "__main__":
    main()
